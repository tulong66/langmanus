#!/usr/bin/env python3
"""
Replicateを使用したQwen2.5-Omniモデルのブラウザエージェント統合テストスクリプト
"""

import os
import sys
import logging
from dotenv import load_dotenv

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# .envファイルを読み込む
load_dotenv()

# srcディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.llm import get_llm_by_type
from src.agents.browser_agent import BrowserAgent
from src.tools.browser import BrowserTools

def test_browser_agent():
    """ブラウザエージェントのテスト"""
    logger.info("ブラウザエージェントのテストを開始します")
    
    # ビジョンLLMを取得（Qwen2.5-Omni）
    vision_llm = get_llm_by_type("vision")
    
    # ブラウザツールを作成
    browser_tools = BrowserTools()
    
    # ブラウザエージェントを作成
    browser_agent = BrowserAgent(
        llm=vision_llm,
        tools=browser_tools.get_tools(),
    )
    
    # 簡単なタスクを実行
    task = "Go to https://www.example.com and tell me what you see on the page."
    
    logger.info(f"タスク: {task}")
    result = browser_agent.run(task)
    
    logger.info(f"結果: {result}")
    
    return result

def main():
    """メイン関数"""
    logger.info("Qwen2.5-Omniモデルを使用したブラウザエージェントのテストを開始します")
    
    # ブラウザエージェントのテスト
    test_browser_agent()
    
    logger.info("テストが完了しました")

if __name__ == "__main__":
    main()
