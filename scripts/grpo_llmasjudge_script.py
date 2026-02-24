# based on gspo_script.py in huggingface/trl, with modifications for gradient regularization and LLMJudge integration

"""
# run with gradient regularization
CUDA_VISIBLE_DEVICES=0,1,4,5 uv run accelerate launch \
    --config_file deepspeed_zero2.yaml \
    grpo_llmasjudge_script.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir grpo-Qwen2.5-0.5B-Instruct_256Judge_lr${lr}_gradreg1e-3 \
    --learning_rate 5e-6 \
    --lr_scheduler_type=constant \
    --dtype bfloat16 \
    --max_completion_length 768 \
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
    --run_name=grpo-Qwen2.5-0.5B-Instruct_udge_gradreg1e-3 \
    --grad_reg_strength 1e-3 \
    --report_to=wandb \
    --project=huggingface \
    --logging_steps=1 \
    --max_steps=1000 

# run without gradient regularization
uv run accelerate launch \
    --config_file deepspeed_zero2.yaml \
    grpo_llmasjudge_script.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --output_dir grpo-Qwen2.5-0.5B-Instruct_Judge_nogradreg \
    --learning_rate 3e-6 \
    --lr_scheduler_type=constant \
    --dtype bfloat16 \
    --max_completion_length 768 \
    --per_device_train_batch_size 16 \
    --scale_rewards=none \
    --num_generations 8 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --beta 0.0 \
    --use_vllm=True \
    --vllm_mode=colocate \
    --vllm_gpu_memory_utilization=0.2 \
    --loss_type grpo \
    --gradient_accumulation_steps 4 \
    --steps_per_generation 1 \
    --run_name=grpo-Qwen2.5-0.5B-Instruct_Judge_nogradreg \
    --report_to=wandb \
    --project=huggingface \
    --logging_steps=1 \
    --grad_reg_strength 0.0 \
    --max_steps=2000 \
    --save_steps=500
"""

import torch

from trl import (
    ModelConfig,
    TrlParser,
)
from trl_gradientregularization import GRPOGradRegConfig, GRPOTrainerGradreg

from LLMJudge import LLMJudge
from reasoning_data_rewards import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    get_gsm8k_questions,
)


if __name__ == "__main__":
    parser = TrlParser((GRPOGradRegConfig, ModelConfig)) # type: ignore
    training_args, model_args = parser.parse_args_and_config()

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation='flash_attention_2',
        dtype=dtype,
    )

    train_dataset = get_gsm8k_questions(split="train")
    eval_dataset = get_gsm8k_questions(split="test")
    max_prompt_length = 256

    llm_judge = LLMJudge(
        model_name='Qwen/Qwen2.5-1.5B-Instruct',
        problem_len=256,
        answer_len=training_args.max_completion_length,
        judging_len=64
    )

    rewards_train = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        llm_judge.score,
        correctness_reward_func,
    ]
    reward_weights = [  # correctness reward only used for logging
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0
    ]
    training_args.reward_weights = reward_weights

    ################
    # Training
    ################
    use_gradreg = training_args.grad_reg_strength != 0.0
    print(f"Using gradient regularization: {use_gradreg} (grad_reg_strength={training_args.grad_reg_strength})")

    trainer = GRPOTrainerGradreg(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=rewards_train,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print("starting train")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    print("train done")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
