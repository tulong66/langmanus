"""
ブラウザエージェントのテストスクリプト
Qwen2.5-Omniモデルを使用したブラウザエージェントの動作をテストする
"""

import logging
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# 環境変数を読み込む
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# ブラウザツールを直接テスト
from src.tools.browser import browser_tool

def test_browser_agent():
    """ブラウザエージェントのテスト"""
    print("=== ブラウザエージェントのテスト開始 ===")

    # テスト用のクエリ
    query = "ブラウザを使って、Googleで「Qwen2.5-Omni」を検索し、最初のページのタイトルを教えてください。"

    # ブラウザツールを直接実行
    print(f"\nクエリ: {query}")
    try:
        result = browser_tool.run(query)
        print(f"\n=== 実行結果 ===")
        print(result)
    except Exception as e:
        print(f"\n=== エラー ===")
        print(f"Error: {str(e)}")

    print("\n=== ブラウザエージェントのテスト終了 ===")

if __name__ == "__main__":
    test_browser_agent()
