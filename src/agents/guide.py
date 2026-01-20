# src/agents/guide.py

from typing import Any, Dict, List, Literal, Optional

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
    scientific_hypothesis: str = Field(
        description="基于当前数据的物理分析与推测。例如：'在 h=0.5 附近磁化率出现峰值，暗示可能存在相变'。"
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
你的核心任务是**基于物理直觉推导假设**并决定后续实验。

### 科学推理要求：
1. **模式识别**：观察数据趋势（如能隙变小、关联函数衰减变慢、序参量突变）。
2. **提出假设**：如果发现异常或剧变，提出物理假设（如“此处可能跨越临界点”、“数值不收敛可能源于能隙关闭”）。
3. **主动加密**：在假设的临界点或感兴趣区域，主动要求**加密采样**（Refined Sampling），而不是仅按初始计划运行。
4. **指标建议**：如果当前指标不足以确认假设，建议计算新的物理量（如纠缠熵 $S_{EE}$、关联长度 $xi$）。
""",
        ),
        MessagesPlaceholder(variable_name="planning_history"),
        (
            "human",
            """### 1. 原始任务: {task_description}
### 2. 当前计划状态: {current_plan}
### 3. 数据与验证摘要:
- 结果摘要: {data_summary}
- Validator 反馈: {validator_feedback}

请进行物理推导并给出决策：
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
    research_log: Optional[List[str]] = None,
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
                "research_log": "\n".join(research_log or []),
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
