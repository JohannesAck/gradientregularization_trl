# Based on https://github.com/vwxyzjn/summarize_from_feedback_details with some modifications.

import os

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PretrainedConfig

def disable_dropout(model: torch.nn.Module):
    """Disable dropout in a model."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = "EleutherAI/pythia-160m",
        base_config: PretrainedConfig = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        if base_config is None:
            base_config = AutoConfig.from_pretrained(base_model)
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            torch.nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def first_true_indices(bools, dtype=torch.long):
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values

def get_reward(model, query_responses, tokenizer, context_length):
    attention_mask = query_responses != tokenizer.pad_token_id
    # position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
    reward_logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )
    sequence_lengths = first_true_indices(query_responses[:, context_length:] == tokenizer.pad_token_id) - 1 + context_length
    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return (
        reward_logits,
        reward_logits[torch.arange(reward_logits.size(0), device=reward_logits.device), sequence_lengths].squeeze(-1),
        sequence_lengths,
    )



def construct_reward_fn(reward_model, rew_tokenizer, max_completion_length, eos_penalty_value, batch_size):
    
    def reward_fn(prompts, completions, **kwargs):
        with torch.no_grad():
            query_token = np.array(kwargs['query_token'])
            response_token = np.array([
                rew_tokenizer.encode(
                    completion + rew_tokenizer.eos_token,
                    # completion,
                    padding="max_length",
                    max_length=max_completion_length,
                    truncation=True,
                ) for completion in completions
            ])
            query_responses = torch.tensor(np.concatenate([query_token, response_token], axis=1)).to(reward_model.device)
            if batch_size is not None:
                scores = []
                for i in range(0, query_responses.shape[0], batch_size):
                    batch = query_responses[i:min(i + batch_size, query_responses.shape[0])]
                    _, batch_scores, _ = get_reward(
                        reward_model, batch, rew_tokenizer, context_length=query_token.shape[1]
                    )
                    scores.append(batch_scores)
                score = torch.cat(scores, dim=0)
            else:
                _, score, _ = get_reward(
                    reward_model, query_responses, rew_tokenizer, context_length=query_token.shape[1]
                )
            # penalty for no eos token
            contain_eos_token = torch.any(query_responses == rew_tokenizer.eos_token_id, dim=-1)
            score = torch.where(contain_eos_token, score, torch.full_like(score, eos_penalty_value))

        return score
    return reward_fn
