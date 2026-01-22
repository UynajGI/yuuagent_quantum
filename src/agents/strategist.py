# src/agents/strategist.py

from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, field_validator


# 1. 定义 Strategist 的输出结构 (保持不变)
class StrategyPlan(BaseModel):
    task_summary: str = Field(description="对用户任务的清晰重述")
    subtasks: List[str] = Field(
        description="按顺序排列的、可由 Programmer 执行的子任务列表"
    )
    required_parameters: Dict[str, Any] = Field(
        description="需要明确的关键参数（如 lattice size, model type, scan range）"
    )
    expected_outputs: List[str] = Field(
        description="最终应产出的数据或文件（如 'mz_vs_h.csv', 'critical_h.txt'）"
    )

    @field_validator("subtasks")
    @classmethod
    def validate_subtasks_not_empty(cls, v):
        if not v or len(v) == 0:
            raise ValueError("subtasks 列表不能为空")
        return v


# 2. 初始化 LLM (保持不变)
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_retries=2,
)


# 3. 构建 Prompt (关键修改点 1)
strategist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子多体物理研究策略专家（Strategist Agent）。
你的任务是为自主实验设定**探索性目标**：
1. **界定探测范围**：设定合理的参数粗扫（Coarse Scan）区间。
2. **定义成功准则**：明确什么样的物理现象（如对称性破缺）标志着任务完成。
3. **预留灵活性**：你的子任务应允许 Guide Agent 根据初步结果进行局部加密采样。
""",
        ),
        MessagesPlaceholder(variable_name="planning_history"),
        (
            "human",
            """用户原始任务：{user_task}
请输出策略计划：{format_instructions}""",
        ),
    ]
)


# 4. 绑定解析器
parser = JsonOutputParser(pydantic_object=StrategyPlan)
chain = strategist_prompt | llm | parser


# 5. 核心函数 (关键修改点 2)
def decompose_task(
    user_task: str, planning_history: list
) -> tuple[Dict[str, Any], list]:
    """
    将用户任务分解为结构化子任务计划。
    实现 Context Quarantine：只接收 planning_history，不接收全局 history。

    Args:
        user_task: 用户输入的自然语言任务
        planning_history: 仅包含规划阶段对话的专用列表

    Returns:
        (Plan, Updated_Planning_History)
    """

    # === Change 2: 仅在 planning_history 中追加记录 ===
    planning_history.append({"role": "user", "content": f"User Task: {user_task}"})

    try:
        result = chain.invoke(
            {
                "user_task": user_task,
                "format_instructions": parser.get_format_instructions(),
                # === Change 3: 显式传递隔离后的历史 ===
                "planning_history": planning_history,
            }
        )
        planning_history.append({"role": "assistant", "content": str(result)})
        return result, planning_history
    except Exception:
        return {"subtasks": []}, planning_history
