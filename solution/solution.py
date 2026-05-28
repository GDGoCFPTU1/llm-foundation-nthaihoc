"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# SDK Imports
from openai import OpenAI
import google.generativeai as genai
import anthropic

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-2.5-flash"
ANTHROPIC_MODEL = "claude-3-5-haiku"

# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    # Sử dụng "mock-key" để tránh crash trên GitHub Actions nếu test suite dùng Mock
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key-openai"))
    
    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.choices[0].message.content
    usage = {
        'input_tokens': response.usage.prompt_tokens,
        'output_tokens': response.usage.completion_tokens
    }
    
    return response_text, latency_seconds, usage

# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "mock-key-gemini"))
    
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens
    )
    
    start_time = time.time()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.text
    usage = {
        'input_tokens': response.usage_metadata.prompt_token_count,
        'output_tokens': response.usage_metadata.candidates_token_count
    }
    
    return response_text, latency_seconds, usage

# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", "mock-key-anthropic"))
    
    start_time = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}]
    )
    latency_seconds = time.time() - start_time
    
    response_text = response.content[0].text
    usage = {
        'input_tokens': response.usage.input_tokens,
        'output_tokens': response.usage.output_tokens
    }
    
    return response_text, latency_seconds, usage

# ---------------------------------------------------------------------------
# Task 4 — Compare Models 
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
        rates = PRICING_1M_TOKENS[model_name]
        return (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000

    results = {}

    resp_4o, lat_4o, usage_4o = call_openai(prompt, model=OPENAI_MODEL)
    results["gpt4o"] = {
        "response": resp_4o,
        "latency": lat_4o,
        "cost": calculate_cost(OPENAI_MODEL, usage_4o["input_tokens"], usage_4o["output_tokens"]),
        "input_tokens": usage_4o["input_tokens"],
        "output_tokens": usage_4o["output_tokens"]
    }

    resp_mini, lat_mini, usage_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    results["gpt4o_mini"] = {
        "response": resp_mini,
        "latency": lat_mini,
        "cost": calculate_cost(OPENAI_MINI_MODEL, usage_mini["input_tokens"], usage_mini["output_tokens"]),
        "input_tokens": usage_mini["input_tokens"],
        "output_tokens": usage_mini["output_tokens"]
    }

    resp_gemini, lat_gemini, usage_gemini = call_gemini(prompt, model=GEMINI_MODEL)
    results["gemini_flash"] = {
        "response": resp_gemini,
        "latency": lat_gemini,
        "cost": calculate_cost(GEMINI_MODEL, usage_gemini["input_tokens"], usage_gemini["output_tokens"]),
        "input_tokens": usage_gemini["input_tokens"],
        "output_tokens": usage_gemini["output_tokens"]
    }

    return results

# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY", "mock-key-gemini"))
    model = genai.GenerativeModel(GEMINI_MODEL)
    chat = model.start_chat(history=[])
    
    print("Welcome to Gemini Chatbot! (type 'quit' or 'exit' to end)")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ['quit', 'exit']:
                print("Bot: Goodbye!")
                break
            
            print("Bot: ", end="", flush=True)
            response = chat.send_message(user_input, stream=True)
            
            for chunk in response:
                print(chunk.text, end="", flush=True)
            print()
            
            if len(chat.history) > 6:
                chat.history = chat.history[-6:]
                
        except (KeyboardInterrupt, EOFError):
            print("\nBot: Exiting chatbot...")
            break
        except Exception as e:
            print(f"\nError: {e}")

# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    delay = base_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                
    raise last_exception

# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    results = []
    for prompt in prompts:
        comparison_result = compare_models(prompt)
        comparison_result["prompt"] = prompt
        results.append(comparison_result)
    return results

# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "|---|---|---|---|---|---|"
    ]
    
    for res in results:
        prompt_text = res["prompt"]
        prompt_trunc = prompt_text[:20] + "..." if len(prompt_text) > 20 else prompt_text
        
        for model_key in ["gpt4o", "gpt4o_mini", "gemini_flash"]:
            data = res[model_key]
            clean_resp = data["response"].replace("\n", " ").strip()
            resp_trunc = clean_resp[:50] + "..." if len(clean_resp) > 50 else clean_resp
            
            lines.append(
                f"| {prompt_trunc} | {model_key} | {resp_trunc} | {data['latency']:.2f}s | "
                f"{data['input_tokens']}/{data['output_tokens']} | ${data['cost']:.6f} |"
            )
            
    return "\n".join(lines)

if __name__ == "__main__":
    pass