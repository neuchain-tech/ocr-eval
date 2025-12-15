#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM(OpenAI互換) に画像+テキストで問い合わせる最小クライアント
- 画像パスはコマンドライン引数
- SYSTEM/USER プロンプトは下の定数を書き換えれば即反映
"""

import argparse
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI
from typing import Optional

# ===== ここを書き換えれば挙動をすぐ変えられます =====
API_BASE = "http://127.0.0.1:8001/v1"  # vLLM を立てたポートに合わせる
# MODEL_NAME = "cyankiwi/Ministral-3-8B-Instruct-2512-AWQ-4bit"  # vLLM起動時の served_model_name
MODEL_NAME = "cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-4bit"

SYSTEM_PROMPT = "あなたは有能な画像アシスタントです。事実に忠実に日本語で簡潔に答えてください。"
USER_TEXT_PROMPT = "この画像の文章をOCRし書かれている文章をすべて抽出してください。回答はマークダウン形式でお願いします。" #図表は図であること、表は表敬式で出力すること。ページ内の位置やレイアウトを乱さないこと"
# USER_TEXT_PROMPT = "入力される画像の左右ページの分割のため、各ページのbboxをJSONで教えて。見開きがない場合は1ページ分のbboxだけで良いです。回答はJSON形式で、それ以外は追加しないようお願いします。"

def image_to_data_url(p: Path) -> str:
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None: mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def detect_served_model(client: OpenAI, override: Optional[str] = None) -> str:
    if override: return override
    models = client.models.list()
    if not models.data:
        raise SystemExit("/v1/models が空です")
    return models.data[0].id

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image_path", type=Path)
    ap.add_argument("--model", default="")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no_stream", action="store_true")
    ap.add_argument("--effort", default="medium", choices=["low","medium","high"])
    args = ap.parse_args()

    img = args.image_path.resolve()
    if not img.exists():
        raise SystemExit(f"画像が見つかりません: {img}")

    client = OpenAI(base_url=API_BASE, api_key="EMPTY")
    model_name = detect_served_model(client, args.model or None)

    data_url = image_to_data_url(img)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": USER_TEXT_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]

    extra = {"reasoning": {"effort": args.effort}}  # ← ここでReasoningを指定

    if not args.no_stream:
        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=True,
            extra_body=extra,  # ← 修正点
        )
        printed_reasoning = False
        printed_answer_header = False
        for chunk in stream:
            delta = chunk.choices[0].delta
            r = getattr(delta, "reasoning_content", None)
            if r:
                if not printed_reasoning:
                    print("=== Reasoning ===")
                    printed_reasoning = True
                print(r, end="", flush=True)
            c = getattr(delta, "content", None)
            if c:
                if printed_reasoning and not printed_answer_header:
                    print("\n\n=== Answer ===")
                    printed_answer_header = True
                print(c, end="", flush=True)
        return

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        extra_body=extra,  # ← 修正点
    )
    msg = resp.choices[0].message
    rc = getattr(msg, "reasoning_content", None)
    ct = getattr(msg, "content", None)
    if rc: print("=== Reasoning ===\n" + rc)
    if ct: print("\n=== Answer ===\n" + ct)
    if not rc and ct is None:
        import json
        print("[DEBUG] Raw response:\n" + json.dumps(resp.model_dump(), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()