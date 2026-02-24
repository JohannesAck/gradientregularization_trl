"""
This file implements gradient regularization (GR) for TRL Trainers (and potentially also general Transformer Trainers).

This class implements the finite-difference GR algorithm by hooking into the `training_step` and `_prepare_inputs` methods.
Instead of a normal optimizer step, we need to pertrub the model weights by the gradient, do a second forward/backward pass at
the perturbed weights.
We then combine the original and perturbed gradients and use them to update the model weights.
Deepspeed and gradient-accumulation make this a bit tricky, requiring us to modify the Accelerator's behavior and write a
custom function to efficiently gather gradients across deepspeed zero partitions.

We make the following implementation choices:
 * We reuse generations from the original weights for efficiency, in principle we would need to generate new actions with the perturbed weights.
 * We do not apply GR to the embedding or output layers, for stability's sake
 * We clip the perturbation as well as both gradients before combining them, avoiding instability from gradient spikes
 * We do perturbations in fp32

The class in this file can be combined with any TRL or Transformers Trainer subclass by inheriting
```
class XTrainerWithGradReg(BaseTrainerGradregDeepSpeed, XTrainer):
    pass
```
or equivalently
```
GRPOTrainerGradreg = type("GRPOTrainerGradreg", (BaseTrainerGradregDeepSpeed, GRPOTrainer), {})
```

We also provide the common trainer as import, such as the GRPOTrainerGradreg class via
```
from trl_gradientregularization import GRPOTrainerGradreg
```

"""


import contextlib
import functools
from typing import Any, Optional, Union

import torch
from torch import nn

from accelerate.utils import DistributedType
from accelerate import Accelerator
from deepspeed import DeepSpeedEngine
import torch.distributed as dist
from deepspeed.utils import safe_get_full_grad, safe_set_full_grad

from trl.trainer.base_trainer import BaseTrainer

from trl_gradientregularization.gradreg_config import TrainArgsGradReg

class AcceleratorManualStepWrapper:

    def __init__(self, accelerator: Accelerator):
        """
        Wrapper for Accelerator to allow manual control of backward and optimizer steps when using DeepSpeed.

        Default Accelerator calls ds_engine.backward, which automatically does an optimizer step.
        This is incompatible with gradient regularization, which needs to modify gradients before the optimizer step.
        We thus provide this wrapper.
        """
        self.acc = accelerator
        

    def backward(self, loss: torch.Tensor, **kwargs: Any) -> None:
        """Perform backpropagation on the given loss tensor without triggering an optimizer step."""
        if self.acc.distributed_type == DistributedType.DEEPSPEED:
            ds_engine: DeepSpeedEngine = self.acc.deepspeed_engine_wrapped.engine
            ds_engine.set_gradient_accumulation_boundary(is_boundary=self.acc.sync_gradients)
            ds_engine.backward(loss, **kwargs)
        else:
            self.acc.backward(loss, **kwargs)
    
    def manual_ds_optimizer_step(self) -> None:
        """Manually trigger the DeepSpeed optimizer step."""
        if self.acc.distributed_type == DistributedType.DEEPSPEED:
            ds_engine: DeepSpeedEngine = self.acc.deepspeed_engine_wrapped.engine
            ds_engine.step()
        else:
            raise RuntimeError("manual_ds_optimizer_step called but not in DeepSpeed mode.")
    
    def manual_ds_zero_grad(self) -> None:
        """Manually zero DeepSpeed optimizer gradients."""
        if self.acc.distributed_type == DistributedType.DEEPSPEED:
            ds_engine: DeepSpeedEngine = self.acc.deepspeed_engine_wrapped.engine
            ds_engine.zero_grad()
        else:
            raise RuntimeError("manual_ds_zero_grad called but not in DeepSpeed mode.")
    
    def manual_ds_refresh_fp32_params(self) -> None:
        """
        Manually refresh DeepSpeed fp32 params from optimizer.
        Not sure if this is actually needed. 
        Could even be bad because it messes up the fp32 master params but who knows.
        """
        if self.acc.distributed_type == DistributedType.DEEPSPEED:
            ds_engine: DeepSpeedEngine = self.acc.deepspeed_engine_wrapped.engine
            ds_engine.optimizer.refresh_fp32_params()
        else:
            raise RuntimeError("manual_ds_refresh_fp32_params called but not in DeepSpeed mode.")
        
    def __getattr__(self, name: str) -> Any:
        """Pass unknown attributes to the wrapped accelerator."""
        return getattr(self.acc, name)

def get_full_model_grads_deepspeed(model: nn.Module, grad_ph: dict[str, torch.Tensor], flat_buff: torch.Tensor, zero_stage: int):
    """
    Collects gradients for models when using deepspeed zero.

    This variant is significantly faster for ZeRO-1/2 to collect gradients than deepspeed's `safe_get_full_grad` in multi-node setups.
    Deepspeed provides the safe_get_full_grad utility, but it converts gradients to fp32 *before* gathering, wasting a lot of bandwidth/time.
    This  might not be an issue in single-node training, but my cluster has 1 GPU per node and a 25GBps interconnnect, soooooo here we are.
    This also lead to me flattening the gradients for the whole model in a single buffer to avoid the otherhead of many small all-gather calls.
    This was helpful on my cluster but may be unnecessary in other setups.

    You might also be asking yourself "zero 1 should not be sharding gradients, why do we need this for zero1?"
    Well the algorithm doesn't but the deepspeed implementation still shards them.
    
    Args:
        model: the model whose gradients we want to collect
        grad_ph: a dict with preallocated tensors to hold the gathered gradients, keyed by parameter name. 
        flat_buff: a preallocated flat buffer to use for all-gather (to avoid per-param allocation overhead)
        zero_stage: the ZeRO optimization stage 
    """
    if zero_stage < 3:  
        idx = 0
        flat_buff.zero_()
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if param._hp_mapping is not None:
                hp_grad_fragment = param._hp_mapping.get_lp_grad_fragment(
                    param._index_in_param_group
                ).flatten()
                lp_frag_address = param._hp_mapping.lp_fragment_address
                reduce_fragment = torch.narrow(flat_buff, 0, idx + lp_frag_address.start, lp_frag_address.numel)
                reduce_fragment.data.copy_(hp_grad_fragment.data)
            idx += param.numel()
        dist.all_reduce(flat_buff, op=dist.ReduceOp.SUM)
        idx = 0
        for pname, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_fragment = flat_buff[idx:idx + param.numel()].reshape_as(param).to(grad_ph[pname].dtype)
            grad_ph[pname].copy_(grad_fragment)
            idx += param.numel()
    elif zero_stage == 3:
        # fallback to slower, default deepspeed method
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_ph[name] = safe_get_full_grad(param)
    return grad_ph


class BaseTrainerGradregDeepSpeed(BaseTrainer):
    """Finite-difference gradient regularization for TRL Trainers with DeepSpeed support.

    This implementation hooks into `training_step` and applies gradient regularization only on
    gradient-accumulation boundaries (`accelerator.sync_gradients == True`), in hopes that it
    stays compatible with upstream Trainer changes.
    """

    _DEFAULT_GRADREG_EXCLUDE = (
        "embed",
        "embedding",
        "lm_head",
        "output",
        "classifier",
        "score",
    )
    _gradreg_capture_inputs = False
    _gradreg_captured_inputs = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_reuse_generations = False
        self.args: TrainArgsGradReg
        self._init_gradreg_hparams_buffers(self.model)
        self.accelerator = AcceleratorManualStepWrapper(self.accelerator) # type: ignore

    def _init_gradreg_hparams_buffers(self, model) -> None:
        """
        Sets up hyperparmeters and buffers for gradient regularization.
        In particular, we create placeholders for the tensors needed to hold the modified and unmodified gradients.
        These are necessary to avoid OOMs later in training, perhaps due to fragmentation? Not sure.
        """
        if self.no_reuse_generations:
            if self.args.steps_per_generation != 1:
                raise AssertionError("steps_per_generation must be 1 when no_reuse_generations is set.")

        grad_reg_exclude = self.args.grad_reg_exclude
        if grad_reg_exclude is None:
            self._gradreg_exclude = self._DEFAULT_GRADREG_EXCLUDE
        elif isinstance(grad_reg_exclude, str):
            self._gradreg_exclude = (grad_reg_exclude,)
        else:
            self._gradreg_exclude = tuple(grad_reg_exclude)

        gradreg_enabled = self.args.grad_reg_strength > 0.0 or callable(self.args.gradreg_scheduler)

        if gradreg_enabled:
            if self.args.grad_reg_eps <= 0.0:
                raise ValueError("grad_reg_eps must be > 0 when gradient regularization is enabled.")

        self._gradreg_buffered_inputs = []
        self._gradreg_capture_inputs = False
        self._gradreg_captured_inputs = None

        self._gradreg_param_names = []
        self.g1_ph = {}
        self.g2_ph = {}
        self.num_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.num_params += param.numel()
                prefix = 'module.' if self.accelerator.distributed_type == DistributedType.DEEPSPEED else ''
                grad_dtype = torch.float32 if self.args.fp32_gradreg else param.dtype
                self.g1_ph[prefix + name] = torch.zeros_like(param, dtype=grad_dtype, device=self.accelerator.device)
                self.g2_ph[prefix + name] = torch.zeros_like(param, dtype=grad_dtype, device=self.accelerator.device)
            if param.requires_grad and not any(substr in name for substr in self._gradreg_exclude):
                self._gradreg_param_names.append(prefix + name)

        self.flat_grad_buffer = torch.zeros(self.num_params, dtype=torch.bfloat16, device=self.accelerator.device)

    @staticmethod
    def _gradreg_clip_(grads: dict[str, torch.Tensor], max_norm: float) -> Optional[float]:
        if max_norm <= 0 or not grads:
            raise ValueError("max_norm must be > 0 and grads must be non-empty for grad clipping.")
        total = torch.zeros((), device=next(iter(grads.values())).device, dtype=torch.float32)
        for grad in grads.values():
            total += grad.float().pow(2).sum()
        norm = torch.sqrt(total)
        clip_coef = max_norm / (norm + 1e-6)
        if clip_coef < 1.0:
            for name in grads:
                grads[name].mul_(clip_coef)
        return norm.item()

    @torch.no_grad()
    def _gradreg_perturb_params_(
        self, model: nn.Module, grads: dict[str, torch.Tensor], eps: float, param_names: list[str]
    ) -> None:
        for name, param in model.named_parameters():
            if name not in param_names:
                continue
            grad = grads[name]
            if self.args.fp32_gradreg:
                param_fp32 = param.data.float()
                param_fp32.add_(grad, alpha=eps)
                param.data.copy_(param_fp32.to(param.dtype))
            else:
                param.data.add_(grad, alpha=eps)
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.manual_ds_refresh_fp32_params()

    def _gradreg_collect_grads(self, model: nn.Module, grad_ph) -> dict[str, torch.Tensor]:
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            grads = get_full_model_grads_deepspeed(
                model,
                grad_ph, 
                self.flat_grad_buffer, 
                zero_stage=self.accelerator.deepspeed_config['zero_optimization']['stage'],
            )
        else:
            grads: dict[str, torch.Tensor] = {}
            for name, param in model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                grad_dtype = torch.float32 if self.args.fp32_gradreg else param.grad.dtype
                grads[name] = param.grad.detach().to(dtype=grad_dtype)
        return grads

    def _gradreg_backward_on_prepared_inputs(
        self,
        model: nn.Module,
        prepared_inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        model.train()
        # if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

        inputs = dict(prepared_inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
            loss = loss / self.current_gradient_accumulation_steps

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.backward(loss, scale_wrt_gas=False)
        else:
            self.accelerator.backward(loss)
        return loss.detach()

    def _gradreg_apply(
        self,
        model: nn.Module,
        prepared_inputs_list: list[dict[str, Union[torch.Tensor, Any]]],
        num_items_in_batch: Optional[torch.Tensor],
        strength: float,
        eps: float,
    ) -> None:
        g1 = self._gradreg_collect_grads(model, self.g1_ph)
        self.orig_param_ph = {
            name: param.detach().clone() for name, param in model.named_parameters()
        }
        g1_norm = self._gradreg_clip_(g1, self.args.grad_reg_g1_clip)

        self._gradreg_perturb_params_(model, g1, eps, self._gradreg_param_names)

        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.manual_ds_zero_grad()
        else:
            model.zero_grad()
        
        if self.no_reuse_generations:
            # clear cached inputs to avoid reusing generations
            if self.use_vllm:
                self._move_model_to_vllm()
            prepared_inputs = super()._prepare_inputs(self._captured_raw_inputs)


        for idx, prepared_inputs in enumerate(prepared_inputs_list):

            do_sync = idx == len(prepared_inputs_list) - 1
            self.accelerator.gradient_state._set_sync_gradients(do_sync)
            context = (
                functools.partial(self.accelerator.no_sync, model=model)
                if not do_sync and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                else contextlib.nullcontext
            )
            with context():
                self._gradreg_backward_on_prepared_inputs(model, prepared_inputs, num_items_in_batch)

        g2 = self._gradreg_collect_grads(model, self.g2_ph)
        g2_norm = self._gradreg_clip_(g2, self.args.grad_reg_g2_clip)

        # restore original parameters
        for name, param in model.named_parameters():
            param.data.copy_(self.orig_param_ph[name].data)
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.manual_ds_refresh_fp32_params()

        named_params = dict(model.named_parameters())
        for name, g1_p in g1.items():
            if name not in self._gradreg_param_names:
                continue
            param = named_params[name]
            g2_p = g2[name]
            combined = g1_p.add_(g2_p - g1_p, alpha=strength / eps)
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                # DeepSpeed requires special handling to set gradients
                safe_set_full_grad(param, combined)
            else:
                param.grad.data.copy_(combined.to(dtype=param.grad.dtype))

        self.accelerator.gradient_state._set_sync_gradients(True)

        mode = "train" if model.training else "eval"
        self._metrics.setdefault(mode, {}).setdefault("grad_reg/g1_norm", []).append(g1_norm)
        self._metrics.setdefault(mode, {}).setdefault("grad_reg/g2_norm", []).append(g2_norm)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]):
        prepared = super()._prepare_inputs(inputs)
        if self._gradreg_capture_inputs:
            self._captured_raw_inputs = inputs
            self._gradreg_captured_inputs = prepared
        return prepared

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # self._init_gradreg_hparams(model)
        strength = (
            self.args.gradreg_scheduler(self.state.global_step)
            if callable(self.args.gradreg_scheduler)
            else self.args.grad_reg_strength
        )
        gradreg_active = strength > 0.0 and self.state.global_step >= self.args.grad_reg_warmup

        self._gradreg_capture_inputs = gradreg_active
        self._gradreg_captured_inputs = None
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._gradreg_capture_inputs = False
    
        if gradreg_active:
            self._gradreg_buffered_inputs.append(self._gradreg_captured_inputs)
        else:
            self._gradreg_buffered_inputs.clear()

        if not self.accelerator.sync_gradients:
            return loss

        # Sync step only: either apply grad-reg or just log g1 norm.
        if gradreg_active:
            self._gradreg_apply(
                model,
                self._gradreg_buffered_inputs,
                num_items_in_batch,
                strength,
                self.args.grad_reg_eps,
            )
            self._gradreg_buffered_inputs.clear()
        else:
            g1_norm = self._gradreg_clip_(self._gradreg_collect_grads(model, self.g1_ph), float("inf"))
            mode = "train" if model.training else "eval"
            self._metrics.setdefault(mode, {}).setdefault("grad_reg/g1_norm", []).append(g1_norm)
        
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            self.accelerator.manual_ds_optimizer_step()
        # if not using DeepSpeed, the optimizer step will be called by the Trainer loop as usual.

        return loss
