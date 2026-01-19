# src/agents/aggregator.py

import json
from typing import Any, Dict, List, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 2. 定义 Aggregator 的输出结构（确保 LLM 返回可解析 JSON）
class AggregationResult(BaseModel):
    summary: str = Field(description="简明的科学总结")
    key_findings: List[str] = Field(description="关键发现列表")
    convergence_status: str = Field(
        description="收敛状态: 'converged', 'not_converged', 'unknown'"
    )
    data_ready_for_visualization: bool = Field(description="是否准备好用于绘图")
    recommendations: List[str] = Field(
        description="给 Conductor 的建议，如是否需要重跑"
    )


# 3. 初始化 LLM（temperature=0 确保确定性）
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)

# 4. 构建 Prompt（注入角色 + 任务上下文）
aggregator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子模拟数据分析专家（Aggregator Agent）。你的任务是：
- 分析来自多个模拟运行的数值结果（如能量、磁化率、动力学轨迹等）
- 判断数值是否收敛（例如通过 MAE、参数敏感性等）
- 提取关键物理结论
- 明确指出数据是否可用于生成出版级图表

请严格基于提供的数据回答，不要编造未出现的结果。
""",
        ),
        (
            "human",
            """研究任务：{task_description}

模拟输出摘要：
{simulation_outputs}

请按以下 JSON Schema 输出分析结果：
{format_instructions}""",
        ),
    ]
)


# 5. 绑定输出解析器
parser = JsonOutputParser(pydantic_object=AggregationResult)
chain = aggregator_prompt | llm | parser


# 6. 核心函数：供 Workflow 调用
def aggregate_simulation_results(
    task_description: str, simulation_outputs: Union[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    聚合并分析模拟结果

    Args:
        task_description: 用户原始任务描述（如 "Run DMRG simulation of 2D Ising model..."）
        simulation_outputs: Executor 生成的结果摘要，可以是：
            - JSON 字符串
            - 字典列表，每个包含 'params', 'output_file', 'metrics' 等字段

    Returns:
        结构化分析结果（dict），符合 AggregationResult schema
    """
    # 确保输入为字符串（LLM 需要）
    if isinstance(simulation_outputs, list):
        input_str = json.dumps(simulation_outputs, indent=2, ensure_ascii=False)
    else:
        input_str = str(simulation_outputs)

    try:
        result = chain.invoke(
            {
                "task_description": task_description,
                "simulation_outputs": input_str,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        # 安全回退：返回结构化错误
        return {
            "summary": f"Aggregation failed due to: {str(e)}",
            "key_findings": [],
            "convergence_status": "unknown",
            "data_ready_for_visualization": False,
            "recommendations": ["Retry aggregation with simplified output."],
        }
