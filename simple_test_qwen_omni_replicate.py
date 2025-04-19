#!/usr/bin/env python3
"""
Replicateを使用したQwen2.5-Omniモデルの単体テストスクリプト
"""

import os
import logging
from dotenv import load_dotenv
from langchain_community.llms import Replicate
from langchain_core.messages import HumanMessage, SystemMessage

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .envファイルを読み込む
load_dotenv()

def test_qwen_omni():
    """Qwen2.5-Omniモデルのテスト"""
    logger.info("Qwen2.5-Omniモデルのテストを開始します")

    # 環境変数から設定を取得
    api_key = os.getenv("VL_API_KEY")
    model = os.getenv("VL_MODEL")

    logger.info(f"モデル: {model}")

    # Replicateモデルを作成
    replicate_llm = Replicate(
        model=model + ":0ca8160f7aaf85703a6aac282d6c79aa64d3541b239fa4c5c1688b10cb1faef1",
        replicate_api_token=api_key,
    )

    # テキスト生成のテスト
    prompt = "What can you do?"
    system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

    logger.info("モデルを呼び出します")
    response = replicate_llm.invoke(
        prompt,
        system_prompt=system_prompt,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
    )

    logger.info(f"応答: {response}")

    return response

def main():
    """メイン関数"""
    test_qwen_omni()

if __name__ == "__main__":
    main()
