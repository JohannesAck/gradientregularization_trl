# based on gspo_script.py in huggingface/trl, with modifications for gradient regularization and LLMJudge integration

"""

# Run with gradient regularization:
uv run accelerate launch \
    --config_file deepspeed_zero2.yaml \
    grpo_rlhf_script.py \
    --model_name_or_path johannesack/Qwen2.5-0.5B_alpaca_sft \
    --output_dir rlhf_gradreg1e-2 \
    --learning_rate 5e-6 \
    --lr_scheduler_type=constant \
    --dtype bfloat16 \
    --max_completion_length 106 \
    --per_device_train_batch_size 16 \
    --scale_rewards=none \
    --num_generations 8 \
    --beta 0.0 \
    --use_vllm=True \
    --vllm_mode=colocate \
    --vllm_gpu_memory_utilization=0.2 \
    --loss_type grpo \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 1 \
    --report_to=wandb \
    --run_name=rlhf_gradreg1e-2  \
    --grad_reg_strength 1e-2 \
    --project=huggingface \
    --logging_steps=1 \
    --max_steps=250 

# Run with gradient KL penalty:
uv run accelerate launch \
    --config_file deepspeed_zero2.yaml \
    grpo_rlhf_script.py \
    --model_name_or_path johannesack/Qwen2.5-0.5B_alpaca_sft \
    --output_dir rlhf_kl \
    --learning_rate 3e-6 \
    --lr_scheduler_type=constant \
    --dtype bfloat16 \
    --max_completion_length 106 \
    --per_device_train_batch_size 8 \
    --scale_rewards=none \
    --num_generations 8 \
    --beta 0.1 \
    --use_vllm=True \
    --vllm_mode=colocate \
    --vllm_gpu_memory_utilization=0.3 \
    --loss_type grpo \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 1 \
    --report_to=wandb \
    --run_name=rlhf_kl  \
    --grad_reg_strength 0.0 \
    --project=huggingface \
    --logging_steps=1 \
    --max_steps=250 
"""

import os

from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from trl import (
    ModelConfig,
    TrlParser,
)
from trl_gradientregularization import GRPOGradRegConfig, GRPOTrainerGradreg

from RewardModel import ScalarModel, construct_reward_fn, disable_dropout


def main():
    parser = TrlParser((GRPOGradRegConfig, ModelConfig)) # type: ignore
    training_args, model_args = parser.parse_args_and_config()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    reward_device = torch.device(f"cuda:{local_rank}")

    query_dataset = 'johannesack/alpaca_qwen_sft1738158738'
    rm_path = 'johannesack/Qwen2.5-0.5B_alpaca_gpt4_1_Nano_rm'
    assert 'Qwen2.5' in model_args.model_name_or_path, "The used dataset is pretokenized for Qwen2.5, please change the dataset for other models."
    
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        # attn_implementation=model_args.attn_implementation,
        attn_implementation='flash_attention_2',
        dtype=dtype,
    )
    
    # set up reward model
    reward_model = ScalarModel.from_pretrained(
        rm_path,
        trust_remote_code=True,
        model_type=ScalarModel
    )
    disable_dropout(reward_model)
    reward_model.to(torch.bfloat16) # type: ignore
    reward_model.to(reward_device) # type: ignore
    reward_model.eval()
    rew_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    rew_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    reward_fn = construct_reward_fn(reward_model, rew_tokenizer, training_args.max_completion_length, -1, 32)

    train_dataset = load_dataset(query_dataset, split="train")
    train_dataset = train_dataset.rename_column('query', 'prompt')
    eval_dataset = load_dataset(query_dataset, split="validation")
    eval_dataset = eval_dataset.with_format("torch", columns=["query", "query_token", "reference_response", "reference_response_token"])
    eval_dataset = eval_dataset.rename_column('query', 'prompt')


    ################
    # Training
    ################
    use_gradreg = training_args.grad_reg_strength != 0.0
    print(f"Using gradient regularization: {use_gradreg} (grad_reg_strength={training_args.grad_reg_strength})")

    trainer = GRPOTrainerGradreg(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[reward_fn],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("starting train")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("train done")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()
