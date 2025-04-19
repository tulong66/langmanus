"""
カスタムLLMラッパーモジュール
Qwen2.5-Omniモデルをブラウザエージェントで使用するためのカスタムラッパー
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_community.llms import Replicate
from pydantic import Field

logger = logging.getLogger(__name__)

class QwenOmniWrapper(BaseChatModel):
    """
    Qwen2.5-Omniモデル用のカスタムラッパー
    bind_toolsメソッドをエミュレートし、ブラウザエージェントと互換性を持たせる
    """

    llm: Replicate = Field(..., description="Replicateモデルインスタンス")
    tools: List[BaseTool] = Field(default_factory=list, description="バインドされたツールのリスト")

    def __init__(self, llm: Replicate):
        """初期化"""
        super().__init__(llm=llm)
        self.tools = []

    def bind_tools(self, tools: List[BaseTool]) -> "QwenOmniWrapper":
        """ツールをバインドするメソッド（互換性のため）"""
        self.tools = tools
        return self

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """メッセージを生成する"""
        # メッセージをQwen2.5-Omniの形式に変換
        formatted_messages = self._format_messages(messages)

        # ツール情報を追加
        if self.tools:
            tools_description = self._format_tools(self.tools)
            # システムメッセージにツール情報を追加
            system_found = False
            for i, msg in enumerate(formatted_messages):
                if msg.get("role") == "system":
                    formatted_messages[i]["content"] += f"\n\nYou have access to the following tools:\n{tools_description}"
                    system_found = True
                    break

            # システムメッセージがない場合は追加
            if not system_found:
                formatted_messages.insert(0, {
                    "role": "system",
                    "content": f"You are a helpful assistant with access to the following tools:\n{tools_description}"
                })

        # Replicateモデルを呼び出す
        try:
            logger.debug(f"Sending messages to Qwen2.5-Omni: {formatted_messages}")
            response = self.llm.invoke(formatted_messages)
            logger.debug(f"Received response from Qwen2.5-Omni: {response}")

            # レスポンスを解析
            ai_message = self._parse_response(response)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error calling Qwen2.5-Omni: {e}")
            # エラー時はシンプルなエラーメッセージを返す
            ai_message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """メッセージをQwen2.5-Omniの形式に変換"""
        formatted_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            else:
                # その他のメッセージタイプはユーザーメッセージとして扱う
                formatted_messages.append({"role": "user", "content": str(message.content)})
        return formatted_messages

    def _format_tools(self, tools: List[BaseTool]) -> str:
        """ツール情報をフォーマット"""
        tools_description = ""
        for i, tool in enumerate(tools):
            tools_description += f"{i+1}. {tool.name}: {tool.description}\n"
        return tools_description

    def _parse_response(self, response: str) -> AIMessage:
        """レスポンスを解析"""
        # ツール呼び出しの検出と処理
        # 現時点では単純なテキスト応答として処理
        return AIMessage(content=response)

    @property
    def _llm_type(self) -> str:
        """LLMタイプを返す"""
        return "qwen-omni-wrapper"
