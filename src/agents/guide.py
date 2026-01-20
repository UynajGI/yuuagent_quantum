# src/agents/guide.py

from typing import Any, Dict, Literal, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. 定义 Guide 的输出结构 (保持不变)
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


# 2. 初始化 LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)


# 3. 构建 Prompt (应用 Context Quarantine)
# 关键改动：显式区分 "计划"、"数据摘要" 和 "验证反馈"，而不是混在历史记录里
guide_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子多体模拟的科研向导（Guide Agent）。
你的职责是**监控科研进度**并**决策下一步方向**。

### 决策原则：
1. **依据计划**：检查当前进度是否偏离了 Strategist 制定的计划。
2. **依据验证**：如果 Validator 报错（如不收敛），必须建议调整参数（如增加 bond dimension）。
3. **依据收敛性**：只有在 Validator 确认收敛后，才建议推进到下一阶段（如 Aggregation）。
4. **隔离细节**：你不需要关注 Python 代码实现，只关注物理参数和结果摘要。

请基于提供的摘要信息，输出结构化建议。
""",
        ),
        # 这里的 planning_history 仅包含 Strategist 和 Guide 的对话，不包含 Programmer 的代码
        MessagesPlaceholder(variable_name="planning_history"),
        (
            "human",
            """### 1. 原始任务
{task_description}

### 2. 当前计划状态 (Plan Context)
{current_plan}

### 3. 数据与验证摘要 (Data Quarantine)
- **数值结果摘要**: {data_summary}
- **Validator 反馈**: {validator_feedback}

请基于以上高层信息，决定下一步操作：
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
    data_summary: str,
    planning_history: list,
    current_plan: Optional[Dict[str, Any]] = None,
    validator_feedback: Optional[str] = None,
) -> tuple[Dict[str, Any], list]:
    """
    根据当前结果建议下一步，实现 Context Quarantine。

    Args:
        task_description: 原始任务
        data_summary: 也就是之前的 current_results，必须是清洗过的摘要字符串
        planning_history: **专用**的规划历史列表（不包含 Programmer/Executor 的日志）
        current_plan: Strategist 生成的计划字典
        validator_feedback: Validator 的校验结果（如 "Energy not converged"）

    Returns:
        (决策结果字典, 更新后的 planning_history)
    """

    # 构造本次输入的上下文描述
    # 这里的 prompt_context 仅用于记录到历史中，保持历史的语义连贯
    context_msg = f"Results Summary: {data_summary}"
    if validator_feedback:
        context_msg += f"\nValidation Issue: {validator_feedback}"

    # 将本次状态作为 User 消息加入规划历史
    # 注意：这里我们模拟了一个持续的对话流，但只包含高层信息
    planning_history.append(
        {
            "role": "user",
            "content": f"Status Update:\n{context_msg}\n\nWhat is the next step based on the plan?",
        }
    )

    try:
        # 调用 LLM，传入隔离后的参数
        result = chain.invoke(
            {
                "task_description": task_description,
                "planning_history": planning_history,
                "current_plan": str(current_plan.get("subtasks", []))
                if current_plan
                else "No explicit plan",
                "data_summary": data_summary,
                "validator_feedback": validator_feedback or "Pass (No issues)",
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # 将决策结果加入历史，形成闭环
        planning_history.append({"role": "assistant", "content": str(result)})
        return result, planning_history

    except Exception as e:
        # 发生错误时的回退策略
        fallback_decision = {
            "next_step": "request_human_help",
            "reasoning": f"Guide encountered an error: {str(e)}",
            "confidence_level": 0.0,
        }
        return fallback_decision, planning_history
