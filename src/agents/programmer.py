# src/agents/programmer.py

from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

# === 关键改动：引入知识加载器 ===
from src.knowledge.loader import get_tenpy_context


# 1. 定义输出结构
class GeneratedCode(BaseModel):
    code: str = Field(description="完整的、可运行的 Python 脚本（必须使用 TeNPy）")
    expected_output_files: list[str] = Field(
        description="脚本将生成的文件列表，必须包含 'results.json'",
        default=["results.json"],
    )
    explanation: str = Field(description="代码逻辑简述")


# 2. 初始化 LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,  # 代码生成必须严谨
    max_retries=2,
)

# 3. 构建 Prompt (应用论文中的 Context Quarantine 策略 )
programmer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个精通 **TeNPy (Tensor Network Python)** 的量子物理编程专家。
你的任务是根据用户需求和提供的参考文档编写 Python 模拟脚本。

### 核心规则：
1. **严格基于文档**：只能使用提供的【参考文档】中存在的 API 和参数。严禁编造函数（Hallucination）。
2. **数据持久化**：模拟结束后，**必须**将关键物理量（如能量、序参量）保存为 `results.json` 文件。
3. **无图模式**：不要调用 `plt.show()` 或生成图片。只计算并保存数据。
4. **自包含**：代码必须包含 `if __name__ == "__main__":` 块，且包含所有必要的 import。

### TeNPy 最佳实践：
- 模型初始化：使用 `tenpy.models` 下的标准模型或 `CouplingMPOModel`。
- 算法引擎：使用 `tenpy.algorithms.dmrg` 或 `tebd`/`tdvp`。
- 参数配置：务必在 `trunc_params` 中设置 `chi_max` (bond dimension)。
""",
        ),
        (
            "human",
            """### 任务描述
{task_description}

### 上下文与参数建议 (来自 Guide/Strategist)
{context}

### 参考文档 (In-Context Knowledge)
{knowledge_base}

请编写 Python 代码：
{format_instructions}""",
        ),
    ]
)

parser = JsonOutputParser(pydantic_object=GeneratedCode)
chain = programmer_prompt | llm | parser


# 4. 核心函数
def generate_tenpy_code(
    task_description: str, context: Optional[str] = None
) -> Dict[str, Any]:
    """
    生成基于 TeNPy 的模拟代码，自动注入相关知识库。
    """
    # 1. 动态检索知识 (RAG / In-Context Learning)
    # 根据任务描述，从 knowledge 文件夹中提取最相关的 API 和 Examples
    knowledge_base = get_tenpy_context(task_description)

    logger_msg = f"Injecting {len(knowledge_base)} chars of TeNPy documentation."
    print(f"[Programmer] {logger_msg}")

    try:
        result = chain.invoke(
            {
                "task_description": task_description,
                "context": context or "无额外参数建议",
                "knowledge_base": knowledge_base,  # <--- 注入点
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        return {
            "code": f"# Error generating code: {str(e)}",
            "expected_output_files": [],
            "explanation": "Generation failed.",
        }
