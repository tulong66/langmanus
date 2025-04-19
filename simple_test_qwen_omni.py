#!/usr/bin/env python3
"""
OpenAI互換APIを使用したQwen2.5-Omniモデルの単体テストスクリプト
"""

import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
    base_url = os.getenv("VL_BASE_URL", None)  # 設定されていない場合はNone
    model = os.getenv("VL_MODEL")
    
    logger.info(f"モデル: {model}")
    logger.info(f"API URL: {base_url if base_url else 'デフォルト'}")
    
    # OpenAI互換APIを使用するChatOpenAIインスタンスを作成
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )
    
    # テキスト生成のテスト
    messages = [
        SystemMessage(content="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."),
        HumanMessage(content="What can you do?")
    ]
    
    logger.info("モデルを呼び出します")
    response = llm.invoke(messages)
    
    logger.info(f"応答: {response.content}")
    
    return response.content

def main():
    """メイン関数"""
    test_qwen_omni()

if __name__ == "__main__":
    main()
