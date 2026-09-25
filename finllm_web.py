# -*- coding: utf-8 -*-
"""
FinLLM 金融助手（网页版，Gradio）
==================================
基座：Qwen/Qwen2.5-7B-Instruct
LoRA：Hello-Liu0618/FinLLM-dpo

运行：
  python finllm_web.py              # 4-bit 量化（约 8GB 显存，默认）
  python finllm_web.py --full       # 全精度 bf16（约 15GB 显存）
  python finllm_web.py --share      # 生成公网分享链接（本地访问不需要）

依赖：pip install gradio torch transformers peft bitsandbytes accelerate
"""

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse

import torch
from modelscope import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import gradio as gr

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"     # 基座（ModelScope 下载）
LORA = "Hello-Liu0618/FinLLM-dpo"          # LoRA 适配器（HF Hub，走镜像）
SYSTEM_PROMPT = "你是专业的金融问答助手，请用中文准确、清晰、完整地回答用户的问题。"

MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7      # 想要更稳定精确就调低，或 do_sample=False
TOP_P = 0.9


def parse_args():
    p = argparse.ArgumentParser(description="FinLLM 金融助手（网页版）")
    p.add_argument("--full", action="store_true",
                   help="全精度 bf16 加载（约 15GB 显存）；默认 4-bit 量化（约 8GB）")
    p.add_argument("--share", action="store_true", help="生成公网可访问的分享链接")
    p.add_argument("--server-port", type=int, default=7860, help="本地端口，默认 7860")
    return p.parse_args()


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
    args = parse_args()
    tokenizer, model = load_model(full=args.full)

    def respond(message, history):
        # 组装带历史的对话
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in history:
            if isinstance(turn, dict):          # Gradio 5.x 的 message dict 格式
                messages.append(turn)
            else:                                # Gradio 4.x 的 [user, assistant] 格式
                messages.append({"role": "user", "content": turn[0]})
                messages.append({"role": "assistant", "content": turn[1]})
        messages.append({"role": "user", "content": message})
        return chat(tokenizer, model, messages)

    demo = gr.ChatInterface(
        fn=respond,
        title="FinLLM 金融助手",
        description="基于 Qwen2.5-7B-Instruct + DPO 对齐的中文金融问答助手",
    )
    demo.launch(server_name="0.0.0.0", server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
