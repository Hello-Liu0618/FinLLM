# -*- coding: utf-8 -*-
"""
FinLLM 本地金融助手（终端对话版）
==================================
基座：Qwen/Qwen2.5-7B-Instruct（4-bit 量化）
LoRA：Hello-Liu0618/FinLLM-dpo（DPO 对齐后）

运行：python finllm_chat.py
需要：CUDA GPU（约 8GB 显存），Linux / WSL 均可
依赖：pip install torch transformers peft bitsandbytes accelerate
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"     # 基座（ModelScope 下载）
LORA = "Hello-Liu0618/FinLLM-dpo"          # LoRA 适配器（HF Hub，走镜像）
SYSTEM_PROMPT = "你是专业的金融问答助手，请用中文准确、清晰、完整地回答用户的问题。"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7      # 想更稳定就把 do_sample 设 False、或把温度调低
TOP_P = 0.9


def load_model(full=False):
    print("从 ModelScope 下载基座 ...")
    model_dir = snapshot_download(BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    if full:
        print("全精度 bf16 加载（约 15GB 显存）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto",
        )
    else:
        print("4-bit 量化加载（约 8GB 显存）...")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map="auto", quantization_config=bnb,
        )

    print(f"加载 LoRA {LORA} ...")
    model = PeftModel.from_pretrained(model, LORA)
    model.eval()
    return tokenizer, model


def chat(tokenizer, model, messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    p = argparse.ArgumentParser(description="FinLLM 金融助手（终端版）")
    p.add_argument("--full", action="store_true",
                   help="全精度 bf16 加载（约 15GB 显存）；默认 4-bit 量化（约 8GB）")
    args = p.parse_args()

    tokenizer, model = load_model(full=args.full)
    print("\nFinLLM 金融助手已就绪！输入问题开始对话，输入 exit / quit 退出。\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("exit", "quit", "退出", "q"):
            break
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        answer = chat(tokenizer, model, messages)
        print(f"\n助手：{answer}\n")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
