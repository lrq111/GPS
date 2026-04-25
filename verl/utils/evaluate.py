"""Compatibility helpers for evaluation and external LLM calls."""

import json

from tqdm import tqdm

from verl.utils.model_api import *  # noqa: F403


def evaluate_with_llm(prompt, data):
    correct = 0
    incorrect = 0

    for _, item in tqdm(enumerate(data), total=len(data), desc="eval", leave=True):
        question = item["question"]
        expected = item["expected"]
        predicted = item["predicted"]
        formatted_prompt = prompt.format(question=question, expected=expected, predicted=predicted)
        response = get_model_response(formatted_prompt)  # noqa: F405
        if response == "Yes":
            correct += 1
            item["eval"] = "true"
        else:
            incorrect += 1
            item["eval"] = "false"

    total = correct + incorrect
    percent = correct / total if total else 0
    print(f"Correct: {correct}, Incorrect: {incorrect}, Percent: {percent}")
    return correct, total, percent, data


if __name__ == "__main__":
    prompt = """
你是一名严格、但能识别同义表达的阅卷老师。请阅读以下信息并判断学生的选择题作答是否正确：

1. 【题目】：
{question}

2. 【正确答案】：
{expected}

3. 【学生的作答】：
{predicted}

你的任务是：
- 首先判断学生的作答是否与正确答案一致（如果含义相同也视为一致）；
- 如果学生作答正确，请只输出：Yes
- 如果学生作答错误，请只输出：No

**重要要求**：
- 不要输出引号、标点、换行、额外文字、空格或其他任何字符。
- 只输出一个单词：Yes 或 No。
    """

    file = "output/xiaobeir1-Qwen2.5-0.5B-Instruct/2025-03-22 22:07:42/evaluation_before_grpo_filtered.json"
    with open(file) as f:
        data = json.load(f)

    correct = 0
    incorrect = 0
    for _, item in tqdm(enumerate(data), total=len(data), desc="eval", leave=True):
        question = item["question"]
        expected = item["expected"]
        predicted = item["predicted"]
        if not predicted:
            continue
        formatted_prompt = prompt.format(question=question, expected=expected, predicted=predicted)
        response = get_model_response(formatted_prompt)  # noqa: F405
        tqdm.write(formatted_prompt)
        tqdm.write(response)
        tqdm.write("-" * 100)
        if response == "Yes":
            correct += 1
        else:
            incorrect += 1

    percent = correct / (correct + incorrect)
    print(f"Correct: {correct}, Incorrect: {incorrect}, Percent: {percent}")
