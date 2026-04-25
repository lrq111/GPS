"""External LLM API helpers shared by training and evaluation."""

from __future__ import annotations

def reasoning_parser(text):
    parts = text.split("</think>", 1)
    if len(parts) > 1:
        return parts[0], parts[1]
    print("没有找到</think>标签, 全部内容为：", parts[0])
    return "", parts[0]


def get_model_response(prompt, model="qwen2.5-72b-instruct", temperature=0.0):
    """Call the default eval LLM and return its raw text response."""
    try:
        return call_qwen_api(prompt, model=model, temperature=temperature)
    except Exception as e:
        print(f"GET MODEL RESPONSE FUNCTION WRONG: {e}")
        return ""


def call_qwen_api(prompt, model="qwen2.5-72b-instruct", temperature=0.0):
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key="sk-xxx",
            base_url="https://xxx",
        )
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"QWEN API RETURN WRONG: {e}")