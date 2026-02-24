import os
import re
from contextlib import contextmanager
try:
    import debugpy
except ImportError:
    class debugpy:
        @staticmethod
        def is_client_connected():
            return False

from trl.data_utils import maybe_apply_chat_template
import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from reasoning_data_rewards import XML_COT_FORMAT


@contextmanager
def _temporary_environ(updates: dict[str, str]):
    """Temporarily set environment variables and restore previous values."""
    old_values: dict[str, str | None] = {k: os.environ.get(k) for k in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _resolve_rank_visible_device(local_rank: int) -> str | None:
    """
    Map LOCAL_RANK to one visible GPU id for child processes.
    Example: CUDA_VISIBLE_DEVICES=4,5,6,7 and LOCAL_RANK=2 -> "6".
    """
    raw_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_visible:
        visible = [item.strip() for item in raw_visible.split(",") if item.strip()]
        if 0 <= local_rank < len(visible):
            return visible[local_rank]
        return None
    return str(local_rank)


JUDGE_PROMPT_GSM8K = """
Judge the correctness of the answer and reasoning for the given problem.
The format is as follows:

<problem>
...
</problem>
<model_answer>
<reasoning>
...
</reasoning>
<answer>
...
</answer>
</model_answer>
<correct_solution>
...
</correct_solution>

You will reply with the following XML format:
<judgement>
...
</judgement>
<correctness_score>
...
</correctness_score>
<coherence_score>
...
</coherence_score>

The model_answer may contain mistakes in the reasoning, the final answer, and in the format.

Give a one sentence judgement on the model_answer, then you will give scores from 1 to 5 for correctness and for coherence of the reasoning trace.
"""

JUDGE_INPUT_FORMAT_GSM8K = """
<problem>
{question}
</problem>
<model_answer>
{model_answer}
</model_answer>
<correct_solution>
{solution}
</correct_solution>
"""

JUDGEMENT_XML_FORMAT_GSM8K = """\
<judgement>
{judgement}
</judgement>
<correctness_score>
{correctness_score}
</correctness_score>
<coherence_score>
{coherence_score}
</coherence_score>
"""


class LLMJudge(object):
    """
    Should've been a very simple LLM judge, but it turned out getting vLLM to start on the right GPU was tricky.
    Currently it simply runs on rank 0 and broadcasts the scores to other ranks, which wastes some compute but could be worse.
    """
    def __init__(
            self,
            model_name: str = 'Qwen/Qwen2.5-1.5B-Instruct',
            problem_len: int = 256,
            answer_len: int = 786,
            judging_len: int = 128,
            dataset: str = 'gsm8k'):
        assert dataset in ['gsm8k', 'math'], "Currently only gsm8k and math are supported."
        self.dataset = dataset
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main_rank = self.rank == 0
        self.model_name = model_name
        self.problem_len = problem_len
        self.answer_len = answer_len
        self.judging_len = judging_len

        # Under torchrun/accelerate each process inherits all visible GPUs; bind to LOCAL_RANK
        # before creating vLLM so judge engines do not all contend for GPU 0.
        local_rank_env = os.environ.get("LOCAL_RANK")
        local_rank = int(local_rank_env) if local_rank_env is not None else None
        self.local_rank = local_rank
        if torch.cuda.is_available() and local_rank is not None:
            try:
                torch.cuda.set_device(local_rank)
            except ValueError:
                pass

        self.llm = None
        self.tokenizer = None
        self.sampling_params = SamplingParams(
            n=1,  
            max_tokens=judging_len,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0,
        )

    def _ensure_local_judge_ready(self) -> None:
        if not self.is_main_rank:
            return
        if self.llm is not None and self.tokenizer is not None:
            return

        judge_visible_device = (
            _resolve_rank_visible_device(self.local_rank)
            if torch.cuda.is_available() and self.local_rank is not None
            else None
        )
        judge_env = {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
        if judge_visible_device is not None:
            judge_env["CUDA_VISIBLE_DEVICES"] = judge_visible_device

        with _temporary_environ(judge_env):
            self.llm = LLM(
                model=self.model_name,
                gpu_memory_utilization=0.15,
                tensor_parallel_size=1,
                distributed_executor_backend="uni",
                # enforce_eager=True,
                enforce_eager=debugpy.is_client_connected(),
                max_model_len=int(1.3 * (self.problem_len + self.answer_len)) + self.judging_len,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        print(f"[LLMJudge][rank={self.rank}] lazy vLLM init done", flush=True)

    def format_input_gsm8k(self, problem: str, answer: str, model_answer: str) -> list[dict]:
        return [{'role': 'system', 'content': JUDGE_PROMPT_GSM8K},
            {'role': 'user', 'content': JUDGE_INPUT_FORMAT_GSM8K.format(
                question="What is the largest single-digit prime number?",
                solution="7",
                model_answer=XML_COT_FORMAT.format(
                   reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
                   answer="7"
                )
            )},
            {'role': 'assistant', 'content': JUDGEMENT_XML_FORMAT_GSM8K.format(
                judgement="The model answer is correct and well-reasoned.",
                correctness_score="5",
                coherence_score="5"
                )},
            {'role': 'user', 'content': JUDGE_INPUT_FORMAT_GSM8K.format(
                question=problem,
                solution=answer,
                model_answer=model_answer
            )}
        ]

    def format_input_math(self, problem: str, answer: str, model_answer: str) -> list[dict]:
        # Placeholder for math dataset formatting
        raise NotImplementedError("Math dataset formatting not implemented yet.")

    def format_input(self, problem: str, answer: str, model_answer: str) -> list[dict]:
        if self.dataset == 'gsm8k':
            return self.format_input_gsm8k(problem, answer, model_answer)
        elif self.dataset == 'math':
            return self.format_input_math(problem, answer, model_answer)
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not supported.")

    def _score_local(self, prompts: list, completions: list, answer: list[str]) -> list[float]:
        self._ensure_local_judge_ready()
        if self.llm is None or self.tokenizer is None:
            raise RuntimeError("Judge model is not initialized on this rank.")

        answers = answer
        inputs = []
        for prompt, completion, answer in zip(prompts, completions, answers):
            input_formatted = self.format_input(
                problem=prompt[-1]['content'],
                answer=answer,
                model_answer=completion[0]['content']
            )
            inputs.append(maybe_apply_chat_template({'prompt': input_formatted}, self.tokenizer)['prompt'])

        responses = self.llm.generate(inputs, sampling_params=self.sampling_params, use_tqdm=False)
        correctness_scores = []
        coherence_scores = []
        sum_scores = []
        for idx in range(len(responses)):
            text = responses[idx].outputs[0].text
            problem = prompts[idx][-1]['content']
            model_answer = completions[idx][0]['content']
            true_answer = answers[idx]
            correctness_score = re.search(r"<correctness_score>\s*(\d+)\s*</correctness_score>", text)
            coherence_score = re.search(r"<coherence_score>\s*(\d+)\s*</coherence_score>", text)
            if correctness_score:
                correctness_scores.append((float(correctness_score.group(1)) - 1) / 4)
            else:
                correctness_scores.append(-1.0)
            if coherence_score:
                coherence_scores.append((float(coherence_score.group(1)) - 1) / 4)
            else:
                coherence_scores.append(-1.0)
            sum_scores.append(correctness_scores[-1] + coherence_scores[-1])
            # if idx == 1:
            #     print('-'*20, f"Question:\n{problem}", f"\n########Answer:\n{true_answer}", f"\n########Model Answer:\n{model_answer}", f"\n########Judge Response:\n{text}", f"\n########Correctness Score: {correctness_scores[-1]}, Coherence Score: {coherence_scores[-1]}, Sum: {sum_scores[-1]}")
        return [s * 2.0 for s in correctness_scores]  # scale to 0-10

    def score(self, prompts: list, completions: list, answer: list[str], **kwargs) -> list[float]:
        if self.world_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return self._score_local(prompts, completions, answer)

        local_payload = {
            "prompts": prompts,
            "completions": completions,
            "answers": answer,
        }
        gathered_payloads = [None] * self.world_size
        dist.all_gather_object(gathered_payloads, local_payload)

        per_rank_scores = None
        if self.is_main_rank:
            per_rank_scores = []
            for payload in gathered_payloads:
                scores = self._score_local(
                    payload["prompts"],
                    payload["completions"],
                    payload["answers"],
                )
                per_rank_scores.append(scores)

        obj_list = [per_rank_scores]
        dist.broadcast_object_list(obj_list, src=0)  
        # could be a gather and scatter instead of all_gather and broadcast, but it doesn't really matter, they're all tiny payloads.
        return obj_list[0][self.rank]
