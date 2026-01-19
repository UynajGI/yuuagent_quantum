# src/agents/executor.py

import json
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. 定义 Executor 的输出结构
class ExecutionResult(BaseModel):
    success: bool = Field(description="是否成功执行")
    output_summary: str = Field(description="数值结果摘要（如能量、磁化率等）")
    metrics: Dict[str, Any] = Field(
        description="关键指标字典，如 {'energy': -25.3, 'mz': 0.85}"
    )
    error_message: Optional[str] = Field(default=None, description="错误信息（如有）")
    output_files: List[str] = Field(description="生成的文件路径列表（如 .txt, .csv）")


# 2. 初始化 LLM（用于解析非结构化输出 → 结构化 JSON）
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)

# 3. Prompt：将原始 stdout/stderr 转为结构化结果
parse_output_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子模拟执行结果解析器。请从以下原始输出中提取关键数值结果，并按 JSON Schema 返回。
只提取明确出现的数字，不要推测或补全。
""",
        ),
        (
            "human",
            """原始输出：
{raw_output}

请严格按以下格式输出：
{format_instructions}""",
        ),
    ]
)

parser = JsonOutputParser(pydantic_object=ExecutionResult)
parse_chain = parse_output_prompt | llm | parser


# 4. 核心函数：执行代码并返回结构化结果
def execute_simulation_code(
    code: str,
    task_description: str,
    timeout: int = 300,  # 5 分钟超时
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行 Programmer 生成的 Python 代码（使用 Renormalizer）

    Args:
        code: 完整可运行的 Python 脚本（字符串）
        task_description: 用户任务描述（用于错误上下文）
        timeout: 执行超时（秒）
        working_dir: 工作目录（默认临时目录）

    Returns:
        结构化执行结果（符合 ExecutionResult schema）
    """
    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="yuuagent_exec_")

    script_path = os.path.join(working_dir, "simulation.py")
    log_path = os.path.join(working_dir, "output.log")

    try:
        # 写入脚本
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        # 执行脚本
        result = subprocess.run(
            ["python", script_path],
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        raw_output = result.stdout + "\n" + result.stderr
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(raw_output)

        if result.returncode == 0:
            # 成功：尝试解析输出
            try:
                # 假设脚本已将结果写入 JSON 文件（最佳实践）
                json_path = os.path.join(working_dir, "results.json")
                if os.path.exists(json_path):
                    with open(json_path, "r") as f:
                        data = json.load(f)
                    return {
                        "success": True,
                        "output_summary": _summarize_metrics(data),
                        "metrics": data,
                        "error_message": None,
                        "output_files": [json_path, log_path],
                    }
                else:
                    # 回退：用 LLM 解析 stdout
                    parsed = parse_chain.invoke(
                        {
                            "raw_output": raw_output[:8000],  # 截断避免超长
                            "format_instructions": parser.get_format_instructions(),
                        }
                    )
                    return parsed
            except Exception as e:
                return {
                    "success": False,
                    "output_summary": "",
                    "metrics": {},
                    "error_message": f"Failed to parse output: {str(e)}",
                    "output_files": [log_path],
                }
        else:
            # 失败
            error_msg = (
                f"Script exited with code {result.returncode}. stderr:\n{result.stderr}"
            )
            return {
                "success": False,
                "output_summary": "",
                "metrics": {},
                "error_message": error_msg,
                "output_files": [log_path],
            }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output_summary": "",
            "metrics": {},
            "error_message": f"Execution timed out after {timeout} seconds",
            "output_files": [],
        }
    except Exception as e:
        return {
            "success": False,
            "output_summary": "",
            "metrics": {},
            "error_message": f"Unexpected error: {str(e)}",
            "output_files": [],
        }


# 5. 辅助函数：将 metrics 转为简短摘要
def _summarize_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.6g}")
    return "; ".join(parts) if parts else "No numerical results found"
