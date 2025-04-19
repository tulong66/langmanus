from langgraph.prebuilt import create_react_agent

from src.prompts import apply_prompt_template
from src.tools import (
    bash_tool,
    browser_tool,
    crawl_tool,
    python_repl_tool,
    tavily_tool,
)

from .llm import get_llm_by_type
from src.config.agents import AGENT_LLM_MAP
from .custom_llm import QwenOmniWrapper

# Create agents using configured LLM types
research_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["researcher"]),
    tools=[tavily_tool, crawl_tool],
    prompt=lambda state: apply_prompt_template("researcher", state),
)

coder_agent = create_react_agent(
    get_llm_by_type(AGENT_LLM_MAP["coder"]),
    tools=[python_repl_tool, bash_tool],
    prompt=lambda state: apply_prompt_template("coder", state),
)

# ブラウザエージェントを作成
# Qwen2.5-Omniモデル用のカスタムラッパーを使用
vl_llm = get_llm_by_type("vision")
qwen_wrapper = QwenOmniWrapper(llm=vl_llm)

# カスタムラッパーを使用してブラウザエージェントを作成
from langchain_core.messages import AIMessage
from typing import Dict, Any

def browser_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """カスタムブラウザエージェント"""
    # ブラウザツールを使用してタスクを実行
    prompt = apply_prompt_template("browser", state)
    last_message = state["messages"][-1]
    instruction = last_message.content if hasattr(last_message, "content") else str(last_message)

    try:
        result = browser_tool.run(instruction)
        return {
            "messages": state["messages"] + [AIMessage(content=result)]
        }
    except Exception as e:
        return {
            "messages": state["messages"] + [AIMessage(content=f"Error executing browser task: {str(e)}")]
        }
