"""
ブラウザツールの直接テストスクリプト
Qwen2.5-Omniモデルを使用したブラウザツールの動作を直接テストする
"""

import logging
import asyncio
import os
from dotenv import load_dotenv
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from langchain_community.llms import Replicate
from langchain_openai import ChatOpenAI

# 環境変数を読み込む
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.DEBUG,  # INFOからDEBUGに変更
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Replicateモデルの設定
from os import getenv
REPLICATE_API_TOKEN = getenv("VL_API_KEY")
VL_MODEL = getenv("VL_MODEL", "lucataco/qwen2.5-omni-7b")

# カスタムラッパークラス
class QwenOmniWrapper:
    """
    Qwen2.5-Omniモデル用のシンプルなラッパー
    """

    def __init__(self, llm):
        self.llm = llm
        self.tools = []
        self._attributes = {}

    def bind_tools(self, tools):
        """ツールをバインドするメソッド（互換性のため）"""
        self.tools = tools
        return self

    def with_structured_output(self, output_schema, **kwargs):
        """構造化出力のためのメソッド（互換性のため）"""
        # 実際には何もしないが、メソッドチェーンのために自身を返す
        return self

    def get(self, key, default=None):
        """辞書のようにアクセスするためのメソッド"""
        return self._attributes.get(key, default)

    def invoke(self, messages):
        """メッセージを生成する"""
        # メッセージをQwen2.5-Omniの形式に変換
        formatted_messages = []

        # デバッグ用にメッセージの型を表示
        logger.debug(f"Message type: {type(messages)}, content: {messages}")

        # メッセージが文字列の場合はユーザーメッセージとして扱う
        if isinstance(messages, str):
            formatted_messages = [{"role": "user", "content": messages}]
        # メッセージがリストの場合は各メッセージを処理
        elif isinstance(messages, list):
            for message in messages:
                if isinstance(message, dict) and "role" in message and "content" in message:
                    # すでに正しい形式の場合はそのまま追加
                    formatted_messages.append(message)
                elif hasattr(message, "type") and hasattr(message, "content"):
                    # LangChainのメッセージオブジェクト
                    if message.type == "system":
                        formatted_messages.append({"role": "system", "content": message.content})
                    elif message.type == "human":
                        formatted_messages.append({"role": "user", "content": message.content})
                    elif message.type == "ai":
                        formatted_messages.append({"role": "assistant", "content": message.content})
                    else:
                        # その他のメッセージタイプはユーザーメッセージとして扱う
                        formatted_messages.append({"role": "user", "content": str(message.content)})
                else:
                    # その他のメッセージタイプはユーザーメッセージとして扱う
                    formatted_messages.append({"role": "user", "content": str(message)})
        else:
            # その他の場合はユーザーメッセージとして扱う
            formatted_messages = [{"role": "user", "content": str(messages)}]

        # メッセージが空の場合はデフォルトメッセージを追加
        if not formatted_messages:
            formatted_messages = [{"role": "user", "content": "Hello"}]

        # Replicateモデルを呼び出す
        try:
            logger.debug(f"Sending messages to Qwen2.5-Omni: {formatted_messages}")
            response = self.llm.invoke(formatted_messages)
            logger.debug(f"Received response from Qwen2.5-Omni: {response}")
            return response
        except Exception as e:
            logger.error(f"Error calling Qwen2.5-Omni: {e}")
            return f"Error: {str(e)}"

    async def ainvoke(self, messages):
        """非同期でメッセージを生成する"""
        # 同期メソッドを呼び出すだけ
        return self.invoke(messages)

def test_browser_direct():
    """ブラウザツールの直接テスト"""
    print("=== ブラウザツールの直接テスト開始 ===")

    # Chromeのパスを設定
    chrome_path = getenv("CHROME_INSTANCE_PATH", "/usr/bin/google-chrome-stable")
    print(f"Using Chrome at: {chrome_path}")

    # ブラウザインスタンスを作成
    browser = Browser(
        config=BrowserConfig(chrome_instance_path=chrome_path)
    )

    # Replicateモデルを作成
    # Replicateモデルは使用しないことにした
    # デバッグ用に環境変数を表示
    logger.debug(f"Replicate API Token: {REPLICATE_API_TOKEN}")
    logger.debug(f"Replicate Model: {VL_MODEL}")

    # カスタムラッパーは使用しないことにした
    # デフォルトのLLMを使用する

    # テスト用のクエリ
    query = "ブラウザを使って、Googleで「Qwen2.5-Omni」を検索し、最初のページのタイトルを教えてください。"

    # BrowserAgentを作成
    # デフォルトのLLMを使用してBrowserAgentを作成
    agent = BrowserAgent(
        task=query,
        browser=browser
    )

    # カスタムラッパーを直接使用するのではなく、デバッグメッセージを表示
    logger.debug(f"Using default LLM for BrowserAgent: {agent.llm}")

    # 非同期実行のためのヘルパー関数
    async def run_agent():
        try:
            result = await agent.run()
            if isinstance(result, AgentHistoryList):
                return result.final_result
            else:
                return str(result)
        except Exception as e:
            return f"Error executing browser task: {str(e)}"

    # 実行
    print(f"\nクエリ: {query}")
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(run_agent())
        print(f"\n=== 実行結果 ===")
        print(result)
    except Exception as e:
        print(f"\n=== エラー ===")
        print(f"Error: {str(e)}")

    print("\n=== ブラウザツールの直接テスト終了 ===")

if __name__ == "__main__":
    test_browser_direct()
