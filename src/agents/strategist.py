# src/agents/strategist.py

from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field, field_validator


# 1. 定义 Strategist 的输出结构
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


# 2. 初始化 LLM（temperature=0 确保确定性）
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)


# 3. 构建 Prompt（注入量子模拟常识）
strategist_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子多体物理研究策略专家（Strategist Agent）。你的任务是：
- 将用户模糊的科研目标分解为具体、可执行的计算子任务
- 明确必须指定的物理参数（如晶格尺寸、模型哈密顿量、算法参数）
- 规划数据产出格式（便于后续聚合和可视化）
- 假设使用 TeNPy 库进行模拟

请输出结构化计划，不要生成代码。
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            """用户原始任务：
{user_task}

请按以下 JSON Schema 输出策略计划：
{format_instructions}""",
        ),
    ]
)


# 4. 绑定解析器
parser = JsonOutputParser(pydantic_object=StrategyPlan)
chain = strategist_prompt | llm | parser


# 5. 核心函数
def decompose_task(user_task: str, history: list) -> tuple[Dict[str, Any], list]:
    """
    将用户任务分解为结构化子任务计划

    Args:
        user_task: 用户输入的自然语言任务（如 "Find phase transition in 2D Ising model"）

    Returns:
        结构化策略计划（符合 StrategyPlan schema）
    """

    history.append({"role": "user", "content": f"User Task: {user_task}"})
    try:
        result = chain.invoke(
            {
                "user_task": user_task,
                "format_instructions": parser.get_format_instructions(),
                "chat_history": history,  # Ensure your prompt template supports this
            }
        )
        history.append({"role": "assistant", "content": str(result)})
        return result, history
    except Exception:
        return {"subtasks": []}, history
