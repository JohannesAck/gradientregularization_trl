# reward functions and data preprocessing based on reward dithering paper code https://openreview.net/forum?id=pG1Y63MqHm ,
# which I believe is based on Will Brown's https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
# which is mostly the same as Open R1 https://github.com/huggingface/open-r1/blob/0e06249d1caa3c1d27a93e590929531291c9c493/src/open_r1/rewards.py#L93

import time
import re

from datasets import Dataset, load_dataset, concatenate_datasets

### GSM8K

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n[\s\S]*?\n</reasoning>\n<answer>\n[\s\S]*?</answer>$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = pattern = pattern = r"^<reasoning>[\s\S]*?</reasoning>[\s\S]*?<answer>[\s\S]*?</answer>$"
    completion_contents = [completion[0]["content"].strip() for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    def count_xml(text) -> float:
        count = 0.0
        if text.count("<reasoning>\n") == 1:
            count += 0.125
        if text.count("\n</reasoning>\n") == 1:
            count += 0.125
        if text.count("\n<answer>\n") == 1:
            count += 0.125
        if text.count("\n</answer>") == 1:
            count += 0.125
            count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
        return count
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")


SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer> 
"""


def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] 
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            {'role': 'assistant', 'content': XML_COT_FORMAT.format(
               reasoning="9 is divisble by 3 and 8 is divisible by 2, but 7 is prime.",
               answer="7"
            )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data
