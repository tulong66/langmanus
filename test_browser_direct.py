"""
ブラウザツールの直接テストスクリプト
Qwen2.5-Omniモデルを使用したブラウザツールの動作を直接テストする
"""

import logging
import asyncio
from dotenv import load_dotenv
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from langchain_community.llms import Replicate

# 環境変数を読み込む
load_dotenv()

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
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
        
    def bind_tools(self, tools):
        """ツールをバインドするメソッド（互換性のため）"""
        return self
    
    def invoke(self, messages):
        """メッセージを生成する"""
        # メッセージをQwen2.5-Omniの形式に変換
        formatted_messages = []
        for message in messages:
            if message.get("role") == "system":
                formatted_messages.append({"role": "system", "content": message["content"]})
            elif message.get("role") == "user":
                formatted_messages.append({"role": "user", "content": message["content"]})
            elif message.get("role") == "assistant":
                formatted_messages.append({"role": "assistant", "content": message["content"]})
            else:
                # その他のメッセージタイプはユーザーメッセージとして扱う
                formatted_messages.append({"role": "user", "content": str(message)})
        
        # Replicateモデルを呼び出す
        try:
            logger.debug(f"Sending messages to Qwen2.5-Omni: {formatted_messages}")
            response = self.llm.invoke(formatted_messages)
            logger.debug(f"Received response from Qwen2.5-Omni: {response}")
            return response
        except Exception as e:
            logger.error(f"Error calling Qwen2.5-Omni: {e}")
            return f"Error: {str(e)}"

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
    replicate_model = Replicate(
        model=VL_MODEL,
        replicate_api_token=REPLICATE_API_TOKEN
    )
    
    # カスタムラッパーを作成
    qwen_wrapper = QwenOmniWrapper(llm=replicate_model)
    
    # テスト用のクエリ
    query = "ブラウザを使って、Googleで「Qwen2.5-Omni」を検索し、最初のページのタイトルを教えてください。"
    
    # BrowserAgentを作成
    agent = BrowserAgent(
        task=query,
        llm=qwen_wrapper,
        browser=browser
    )
    
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
