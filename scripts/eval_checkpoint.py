#!/usr/bin/env python3
"""
Evaluate a saved GRPO checkpoint and write averaged eval metrics to a .txt file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward, think_format_reward
from reasoning_data_rewards import (
    correctness_reward_func,
    int_reward_func,
    strict_format_reward_func,
    soft_format_reward_func,
    xmlcount_reward_func,
    get_gsm8k_questions,
)


NUMINA_SYSTEM_PROMPT = (
    "A conversation between user and assistant. The user asks a question, and the assistant solves it. The "
    "assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
    "The reasoning process and answer are enclosed within <think></think> tags, i.e., <think>\nThis is my "
    "reasoning.\n</think>\nThis is my answer."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model/checkpoint and write results to .txt.")
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="AI-MO/NuminaMath-TIR",
        choices=["AI-MO/NuminaMath-TIR", "numina_math_tir", "gsm8k"],
        help="Dataset setup to use for evaluation.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Path to checkpoint directory. Required unless --model_path is provided.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model path or model id (e.g. Qwen/Qwen3-0.6B). Optional.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory where eval_averaged_results.txt will be written. Defaults to checkpoint_dir.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        help="Dataset split expression.",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=16)
    parser.add_argument("--num_generations_eval", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--vllm_mode", type=str, default="colocate")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2)
    parser.add_argument(
        "--output_filename",
        type=str,
        default="eval_averaged_results.txt",
        help="Output .txt filename inside checkpoint folder.",
    )
    return parser.parse_args()


def make_numina_conversation(example: dict) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": NUMINA_SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }


def build_eval_setup(eval_dataset_name: str, eval_split: str):
    if eval_dataset_name == "gsm8k":
        if eval_split not in {"train", "test"}:
            raise ValueError(
                "For eval_dataset='gsm8k', eval_split must be 'train' or 'test' "
                "because evaluation uses get_gsm8k_questions from reasoning_data_rewards."
            )
        eval_dataset = get_gsm8k_questions(split=eval_split)
        reward_funcs = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
        hf_dataset_name = "openai/gsm8k"
    else:
        eval_dataset = load_dataset("AI-MO/NuminaMath-TIR", split=eval_split)
        eval_dataset = eval_dataset.map(make_numina_conversation)
        eval_dataset = eval_dataset.remove_columns(["messages", "problem"])
        reward_funcs = [think_format_reward, accuracy_reward]
        hf_dataset_name = "AI-MO/NuminaMath-TIR"

    return eval_dataset, reward_funcs, hf_dataset_name


def main() -> None:
    args = parse_args()
    if args.model_path is None and args.checkpoint_dir is None:
        raise ValueError("Provide either --checkpoint_dir or --model_path.")

    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir is not None else None
    if checkpoint_dir is not None and not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    model_path = args.model_path if args.model_path is not None else str(checkpoint_dir)
    output_dir_target = args.output_dir.resolve() if args.output_dir is not None else checkpoint_dir
    if output_dir_target is None:
        raise ValueError("When using --model_path without --checkpoint_dir, you must provide --output_dir.")
    output_dir_target.mkdir(parents=True, exist_ok=True)

    eval_dataset, reward_funcs, hf_dataset_name = build_eval_setup(args.eval_dataset, args.eval_split)

    if checkpoint_dir is not None:
        run_name = f"eval_{checkpoint_dir.parent.name}_{checkpoint_dir.name}"
    else:
        run_name = f"eval_{model_path.replace('/', '_')}"
    tmp_output_dir = output_dir_target / "eval_tmp"

    training_args = GRPOConfig(
        output_dir=str(tmp_output_dir),
        run_name=run_name,
        report_to=[],
        do_train=False,
        do_eval=True,
        eval_strategy="no",
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        num_generations=args.num_generations_eval,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        model_init_kwargs={"dtype": torch.bfloat16},
        seed=args.seed,
    )

    trainer = GRPOTrainer(
        model=model_path,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=None,
        eval_dataset=eval_dataset,
    )

    metrics = trainer.evaluate()
    eval_from_return = {k: v for k, v in metrics.items() if k.startswith("eval_")}
    eval_from_log_history = {}
    for entry in reversed(trainer.state.log_history):
        if any(str(k).startswith("eval_") for k in entry.keys()):
            eval_from_log_history = {k: v for k, v in entry.items() if str(k).startswith("eval_")}
            break
    averaged_metrics = dict(sorted({**eval_from_return, **eval_from_log_history}.items()))

    now = datetime.now(timezone.utc).isoformat()
    out_path = output_dir_target / args.output_filename
    lines = [
        f"timestamp_utc: {now}",
        f"model_path: {model_path}",
        f"output_dir: {output_dir_target}",
        f"eval_dataset: {args.eval_dataset}",
        f"hf_dataset_name: {hf_dataset_name}",
        f"eval_split: {args.eval_split}",
        f"per_device_batch_size: {args.per_device_batch_size}",
        f"num_generations_eval: {args.num_generations_eval}",
        f"max_completion_length: {args.max_completion_length}",
        f"seed: {args.seed}",
        "metrics:",
    ]
    lines.extend([f"{key}: {value}" for key, value in averaged_metrics.items()])
    out_path.write_text("\n".join(lines) + "\n")

    print(f"Wrote averaged eval metrics to: {out_path}")
    for key, value in averaged_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
