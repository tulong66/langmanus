#!/usr/bin/env python3
"""
OpenAI互換APIを使用したQwen2.5-Omniモデルの画像入力テストスクリプト
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .envファイルを読み込む
load_dotenv()

# srcディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from src.agents.custom_llm import QwenOmniVisionWrapper
from langchain_core.messages import SystemMessage

def test_image_understanding(image_path, query):
    """画像理解のテスト"""
    logger.info(f"画像理解のテストを開始します: {image_path}")
    
    # 環境変数から設定を取得
    api_key = os.getenv("VL_API_KEY")
    base_url = os.getenv("VL_BASE_URL", None)  # 設定されていない場合はNone
    model = os.getenv("VL_MODEL")
    
    # OpenAI互換APIを使用するChatOpenAIインスタンスを作成
    openai_llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    
    # QwenOmniVisionWrapperでラップ
    llm = QwenOmniVisionWrapper(llm=openai_llm)
    
    # 画像を含むメッセージを作成
    image_message = llm.create_image_message(image_path, query)
    
    # システムメッセージを追加
    messages = [
        SystemMessage(content="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."),
        image_message
    ]
    
    logger.info("モデルを呼び出します")
    response = llm.invoke(messages)
    
    logger.info(f"応答: {response.content}")
    
    return response.content

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description="Qwen2.5-Omniモデルの画像理解テスト")
    parser.add_argument("--image", type=str, required=True, help="画像ファイルのパス")
    parser.add_argument("--query", type=str, default="What do you see in this image?", help="画像に対する質問")
    
    args = parser.parse_args()
    
    logger.info("Qwen2.5-Omniモデルの画像理解テストを開始します")
    
    # 画像理解のテスト
    test_image_understanding(args.image, args.query)
    
    logger.info("テストが完了しました")

if __name__ == "__main__":
    main()
