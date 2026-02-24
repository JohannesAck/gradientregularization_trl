from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class GradRegConfigMixin:
    """Shared gradient-regularization hyperparameters for TRL configs."""

    grad_reg_strength: float = field(default=1e-3, metadata={"help": "Gradient regularization strength."})
    grad_reg_eps: float = field(default=1e-3, metadata={"help": "Finite-difference epsilon."})
    grad_reg_warmup: int = field(default=0, metadata={"help": "Warmup steps before applying grad-reg."})
    grad_reg_g1_clip: float = field(default=10.0, metadata={"help": "Max norm clip for g1 gradients."})
    grad_reg_g2_clip: float = field(default=10.0, metadata={"help": "Max norm clip for g2 gradients."})
    grad_reg_exclude: list[str] | None = field(
        default=None,
        metadata={"help": "Optional list of parameter-name substrings to exclude from grad-reg."},
    )
    fp32_gradreg: bool = field(default=True, metadata={"help": "Run grad-reg math in fp32 for stability."})
    gradreg_scheduler: object = field(default=None, metadata={"help": "Optional callable scheduler for grad-reg."})


TrainArgsGradReg = dataclass(type("TrainArgsGradReg", (GradRegConfigMixin, TrainingArguments), {}))
