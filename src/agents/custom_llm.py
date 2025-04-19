"""
カスタムLLMラッパーモジュール
Qwen2.5-Omniモデルをブラウザエージェントで使用するためのカスタムラッパー
Replicateを使用してQwen2.5-Omniモデルを呼び出す
"""

import logging
from typing import Any, Dict, List, Optional

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
    Replicateを使用してQwen2.5-Omniモデルを呼び出す
    bind_toolsメソッドをサポートし、ブラウザエージェントと互換性を持たせる
    """

    llm: Replicate = Field(..., description="Replicateモデルインスタンス")
    tools: List[BaseTool] = Field(default_factory=list, description="バインドされたツールのリスト")

    def __init__(
        self,
        llm: Replicate
    ):
        """初期化"""
        super().__init__(llm=llm)
        self.tools = []
        self._attributes = {}

    def bind_tools(self, tools: List[BaseTool]) -> "QwenOmniWrapper":
        """ツールをバインドするメソッド（OpenAI互換）"""
        self.tools = tools
        return self

    def with_structured_output(self, output_schema: Any, **kwargs: Any) -> "QwenOmniWrapper":
        """構造化出力のためのメソッド（互換性のため）"""
        return self

    def get(self, key, default=None):
        """辞書のようにアクセスするためのメソッド"""
        return self._attributes.get(key, default)

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

        # ユーザーメッセージとシステムメッセージを抽出
        prompt = ""
        system_prompt = ""

        for msg in formatted_messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
            elif msg.get("role") == "system":
                system_prompt = msg.get("content", "")

        # ツール情報を追加
        if self.tools:
            tools_description = self._format_tools(self.tools)
            if system_prompt:
                system_prompt += f"\n\nYou have access to the following tools:\n{tools_description}"
            else:
                system_prompt = f"You are a helpful assistant with access to the following tools:\n{tools_description}"

        # Replicateモデルを呼び出す
        try:
            logger.debug(f"Sending prompt to Qwen2.5-Omni: {prompt}")
            logger.debug(f"System prompt: {system_prompt}")

            # Replicateのパラメータを設定
            invoke_params = {
                "system_prompt": system_prompt,
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
            }

            # kwargsから追加パラメータを取得
            if "max_tokens" in kwargs:
                invoke_params["max_new_tokens"] = kwargs["max_tokens"]
            if "temperature" in kwargs:
                invoke_params["temperature"] = kwargs["temperature"]

            response = self.llm.invoke(prompt, **invoke_params)
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

        # デバッグ用にメッセージの型を表示
        logger.debug(f"Message type: {type(messages)}, content: {messages}")

        for message in messages:
            if isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif hasattr(message, "content"):
                # content属性を持つメッセージはユーザーメッセージとして扱う
                formatted_messages.append({"role": "user", "content": str(message.content)})
            else:
                # その他のメッセージタイプはユーザーメッセージとして扱う
                formatted_messages.append({"role": "user", "content": str(message)})

        # メッセージが空の場合はデフォルトメッセージを追加
        if not formatted_messages:
            formatted_messages = [{"role": "user", "content": "Hello"}]

        return formatted_messages

    def _format_tools(self, tools: List[BaseTool]) -> str:
        """ツール情報をフォーマット"""
        tools_description = ""
        for i, tool in enumerate(tools):
            tools_description += f"{i+1}. {tool.name}: {tool.description}\n"
        return tools_description

    def _parse_response(self, response: str) -> AIMessage:
        """レスポンスを解析"""
        return AIMessage(content=response)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """非同期でメッセージを生成する"""
        # 同期メソッドを呼び出すだけ
        return self._generate(messages, stop, run_manager, **kwargs)

    @property
    def _llm_type(self) -> str:
        """LLMタイプを返す"""
        return "qwen-omni-wrapper"

