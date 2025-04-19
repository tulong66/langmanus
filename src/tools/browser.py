import asyncio
import logging

from pydantic import BaseModel, Field
from typing import Optional, ClassVar, Type
from langchain.tools import BaseTool
from browser_use import AgentHistoryList, Browser, BrowserConfig
from browser_use import Agent as BrowserAgent
from src.agents.llm import vl_llm
from src.tools.decorators import create_logged_tool
from src.config import CHROME_INSTANCE_PATH

logger = logging.getLogger(__name__)

expected_browser = None

# Use Chrome instance if specified
if CHROME_INSTANCE_PATH:
    expected_browser = Browser(
        config=BrowserConfig(chrome_instance_path=CHROME_INSTANCE_PATH)
    )
    logger.info(f"Using Chrome instance at: {CHROME_INSTANCE_PATH}")
else:
    logger.warning("No Chrome instance path specified. Browser tool may not work properly.")


class BrowserUseInput(BaseModel):
    """Input for WriteFileTool."""

    instruction: str = Field(..., description="The instruction to use browser")


class BrowserTool(BaseTool):
    name: ClassVar[str] = "browser"
    args_schema: Type[BaseModel] = BrowserUseInput
    description: ClassVar[str] = (
        "Use this tool to interact with web browsers. Input should be a natural language description of what you want to do with the browser, such as 'Go to google.com and search for browser-use', or 'Navigate to Reddit and find the top post about AI'."
    )

    _agent: Optional[BrowserAgent] = None

    def _run(self, instruction: str) -> str:
        """Run the browser task synchronously."""
        from src.agents.custom_llm import QwenOmniWrapper

        logger.info(f"Browser tool executing task: {instruction}")
        # カスタムラッパーを作成
        qwen_wrapper = QwenOmniWrapper(llm=vl_llm)

        self._agent = BrowserAgent(
            task=instruction,  # Will be set per request
            llm=qwen_wrapper,  # カスタムラッパーを使用
            browser=expected_browser,
        )
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self._agent.run())
                if isinstance(result, AgentHistoryList):
                    logger.info(f"Browser task completed with result: {result.final_result}")
                    return result.final_result
                else:
                    logger.info(f"Browser task completed with result: {str(result)}")
                    return str(result)
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error executing browser task: {str(e)}")
            return f"Error executing browser task: {str(e)}"

    async def _arun(self, instruction: str) -> str:
        """Run the browser task asynchronously."""
        from src.agents.custom_llm import QwenOmniWrapper

        logger.info(f"Browser tool executing async task: {instruction}")
        # カスタムラッパーを作成
        qwen_wrapper = QwenOmniWrapper(llm=vl_llm)

        self._agent = BrowserAgent(
            task=instruction,  # Will be set per request
            llm=qwen_wrapper,  # カスタムラッパーを使用
            browser=expected_browser,
        )
        try:
            result = await self._agent.run()
            if isinstance(result, AgentHistoryList):
                logger.info(f"Browser async task completed with result: {result.final_result}")
                return result.final_result
            else:
                logger.info(f"Browser async task completed with result: {str(result)}")
                return str(result)
        except Exception as e:
            logger.error(f"Error executing browser async task: {str(e)}")
            return f"Error executing browser task: {str(e)}"


BrowserTool = create_logged_tool(BrowserTool)
browser_tool = BrowserTool()
