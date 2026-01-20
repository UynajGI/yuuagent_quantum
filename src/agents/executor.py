# src/agents/executor.py

import glob
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from src.tools.slurm import check_job_status_core, submit_slurm_job_core


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


def execute_simulation_code(
    code: str,
    task_description: str,
    parameter_grid: Optional[List[Dict[str, Any]]] = None,  # <--- 关键新增接口
    timeout: int = 3600,  # 批处理超时时间通常要长一些
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行模拟。自动判断是本地执行还是 Slurm 批量执行。
    """
    if working_dir is None:
        working_dir = tempfile.mkdtemp(prefix="yuuagent_exec_")

    os.makedirs(working_dir, exist_ok=True)
    script_path = os.path.join(working_dir, "simulation.py")

    # 1. 写入代码文件
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    # === 分支：是否存在参数网格？ ===
    if parameter_grid and len(parameter_grid) > 0:
        print(
            f"[Executor] Detected {len(parameter_grid)} parameters. Switching to Slurm Batch Mode."
        )
        return _execute_batch_slurm(script_path, parameter_grid, timeout)
    else:
        # === 默认：本地单次执行 (保持原有逻辑，或者也可以改为提交单次Slurm) ===
        # 这里为了演示简单，保留之前的 subprocess.run 逻辑，或者调用 submit_slurm_job(..., parameter_grid=None)
        # 建议：如果是在登录节点，最好也用 Slurm 提交单次作业，避免占用登录节点资源
        print("[Executor] No parameters provided. Submitting single Slurm job.")
        return _execute_batch_slurm(script_path, None, timeout)  # 复用轮询逻辑


def _execute_batch_slurm(script_path, parameter_grid, timeout):
    """
    内部辅助函数：处理 Slurm 提交、轮询和结果收集
    """
    working_dir = os.path.dirname(script_path)

    # 1. 提交作业 (复用工具)
    # 注意：submit_slurm_job 是个 Tool，直接调用其 python 函数逻辑即可
    # 如果它是被 @tool 装饰的，可能需要 .invoke 或者直接提取逻辑。
    # 这里假设我们直接调用上面定义的 python 函数逻辑。
    submit_msg = submit_slurm_job_core(script_path, parameter_grid=parameter_grid)

    if "Success" not in submit_msg:
        return {
            "success": False,
            "error_message": submit_msg,
            "metrics": {},
            "output_files": [],
        }

    # 提取 Job ID (简单的字符串处理)
    job_id = submit_msg.split("ID: ")[-1].strip().rstrip(".")
    print(f"[Executor] Job {job_id} submitted. Waiting for completion...")

    # 2. 轮询等待
    start_time = time.time()
    while True:
        # === 关键修改：调用 Core 函数 ===
        status = check_job_status_core(job_id)
        if status == "COMPLETED":
            break

        if time.time() - start_time > timeout:
            return {"success": False, "error_message": "Slurm execution timed out."}

        time.sleep(10)

    # 3. 收集结果
    # 假设 Programmer 生成的代码会将结果保存为 results_{job_id}.json
    # 如果是单次任务，可能是 results.json
    results = []

    if parameter_grid:
        # 批量模式：查找所有 results_*.json
        json_files = glob.glob(os.path.join(working_dir, "results_*.json"))
        # 按索引排序确保顺序
        # 假设文件名是 results_0.json, results_1.json
        try:
            json_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        except ValueError:
            pass  # 如果文件名格式不对就不强求排序
    else:
        # 单次模式
        json_files = glob.glob(os.path.join(working_dir, "results.json"))

    if not json_files:
        # 读取错误日志
        err_files = glob.glob(os.path.join(working_dir, "*.err"))
        err_msg = "No result files found."
        if err_files:
            with open(err_files[0], "r") as f:
                err_msg += f"\nLast Error Log:\n{f.read()}"
        return {
            "success": False,
            "error_message": err_msg,
            "metrics": {},
            "output_files": glob.glob(os.path.join(working_dir, "*")),
        }

    # 读取所有数据
    for jf in json_files:
        with open(jf, "r") as f:
            results.append(json.load(f))

    # 4. 返回
    # 如果是批量，metrics 返回列表；如果是单次，返回字典（保持接口兼容性）
    final_metrics = results if parameter_grid else results[0]

    return {
        "success": True,
        "output_summary": f"Collected {len(results)} results from Slurm.",
        "metrics": final_metrics,  # Aggregator 需要适配处理 List
        "error_message": None,
        "output_files": json_files,
    }
