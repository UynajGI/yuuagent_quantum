# src/agents/guide.py

from typing import Any, Dict, List, Literal, Optional, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. 定义 Guide 的输出结构
class GuideDecision(BaseModel):
    next_step: Literal[
        "run_additional_simulation",
        "increase_bond_dimension",
        "scan_more_parameters",
        "proceed_to_aggregation",
        "request_human_help",
        "terminate_early",
    ] = Field(description="下一步建议动作")
    reasoning: str = Field(description="决策的科学依据")
    suggested_parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="如需新模拟，建议的参数（如 {'h': 2.5, 'M': 128}）"
    )
    confidence_level: float = Field(
        ge=0.0, le=1.0, description="对当前结论的置信度（0~1）"
    )


# 2. 初始化 LLM（temperature=0 确保确定性）
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)


# 3. 构建 Prompt
guide_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子多体模拟的科研向导（Guide Agent）。你的任务是：
- 根据当前已有的模拟结果和原始研究目标，判断是否需要补充计算
- 评估数据是否足以支持物理结论（如相变点、收敛性）
- 建议具体的下一步操作（如增加 bond dim、扫描新参数点）
- 若数据充分且可靠，建议进入聚合（aggregation）阶段

请基于科学严谨性回答，避免过度自信。
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """原始研究任务：{task_description}

当前已获得的结果摘要：
{current_results}

请按以下 JSON Schema 输出你的建议：
{format_instructions}""",
        ),
    ]
)


# 4. 绑定解析器
parser = JsonOutputParser(pydantic_object=GuideDecision)
chain = guide_prompt | llm | parser


# 5. 核心函数：供 Conductor 调用
def guide_next_step(
    task_description: str,
    current_results: Union[str, List[Dict[str, Any]], Dict[str, Any]],
    history: list,
) -> tuple[Dict[str, Any], list]:
    """
    根据当前结果，建议下一步行动

    Args:
        task_description: 用户原始任务（如 "Find critical h in 2D Ising model"）
        current_results: 已有模拟结果，可以是：
            - 字典列表 [{'h': 1.0, 'mz': 0.9}, ...]
            - 单个字典
            - JSON 字符串

    Returns:
        结构化决策（符合 GuideDecision schema）
    """
    history.append(
        {
            "role": "user",
            "content": f"Current simulation results: {current_results}. What is the next step?",
        }
    )
    # 标准化输入为字符串
    try:
        result = chain.invoke(
            {
                "task_description": task_description,
                "current_results": current_results,
                "chat_history": history,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        history.append({"role": "assistant", "content": str(result)})
        return result, history
    except Exception:
        return {"next_step": "request_human_help"}, history
