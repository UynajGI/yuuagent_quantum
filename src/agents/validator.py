# src/agents/validator.py

import json
from typing import Any, Dict, List, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. 定义 Validator 的输出结构
class ValidationReport(BaseModel):
    is_valid: bool = Field(description="结果是否物理合理且数值可靠")
    issues: List[str] = Field(
        description="发现的问题列表（如 'Energy not converged', 'Mz > 0.5'）"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="验证置信度（1.0 = 完全可信）"
    )
    recommendations: List[str] = Field(
        description="改进建议（如 'Increase bond dimension', 'Check Hamiltonian definition'）"
    )


# 2. 初始化 LLM（temperature=0 确保确定性）
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_retries=2,
)


# 3. 构建 Prompt（注入量子多体物理常识）
validator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子多体模拟验证专家（Validator Agent）。你的任务是：
- 检查模拟结果是否满足基本物理约束（如能量下界、序参量范围）
- 评估数值收敛性（如 bond dimension、时间步长是否足够）
- 识别常见错误（如哈密顿量定义错误、边界条件混淆）
- 若结果可疑，明确指出问题并给出改进建议

请基于以下领域知识判断：
- 横场伊辛模型：|Mz| ∈ [0, 0.5]，基态能量应随 h 单调变化
- DMRG 收敛：bond dim 增加时能量变化应 < 1e-6
- 自旋系统：总 Sz 应在 [-L/2, L/2] 范围内

只基于提供的数据判断，不要假设未给出的信息。
""",
        ),
        (
            "human",
            """原始研究任务：{task_description}

模拟结果摘要：
{simulation_results}

请按以下 JSON Schema 输出验证报告：
{format_instructions}""",
        ),
    ]
)


# 4. 绑定解析器
parser = JsonOutputParser(pydantic_object=ValidationReport)
chain = validator_prompt | llm | parser


# 5. 核心函数
def validate_simulation_results(
    task_description: str,
    simulation_results: Union[str, Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    验证模拟结果的物理合理性与数值可靠性

    Args:
        task_description: 用户原始任务
        simulation_results: Executor 返回的结果（含 metrics, output_files 等）

    Returns:
        结构化验证报告（符合 ValidationReport schema）
    """
    # 标准化输入为字符串
    if isinstance(simulation_results, (dict, list)):
        input_str = json.dumps(simulation_results, indent=2, ensure_ascii=False)
    else:
        input_str = str(simulation_results)

    try:
        result = chain.invoke(
            {
                "task_description": task_description,
                "simulation_results": input_str[:8000],  # 防止超长
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        return {
            "is_valid": False,
            "issues": [f"Validation failed due to: {str(e)}"],
            "confidence": 0.0,
            "recommendations": ["Retry validation with simplified input."],
        }
