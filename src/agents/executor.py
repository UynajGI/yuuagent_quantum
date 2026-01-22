# src/agents/executor.py

import glob
import json
import os
import tempfile
import time
from pathlib import Path
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
    parameter_grid: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 3600,
    working_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    执行模拟。统一使用 Slurm Batch 接口，确保参数传递一致性。
    """
    if working_dir is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        workspace_root = project_root / "workspace"
        os.makedirs(workspace_root, exist_ok=True)
        working_dir = tempfile.mkdtemp(prefix="exec_", dir=workspace_root)

    os.makedirs(working_dir, exist_ok=True)
    script_path = os.path.join(working_dir, "simulation.py")

    # 2. 写入代码
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    # === 3. 统一批处理逻辑 (Unified Batch Strategy) ===
    # 标记原本是否为单次运行，用于后续解包结果
    is_single_run = False

    if not parameter_grid or len(parameter_grid) == 0:
        print("[Executor] Single run detected. Converting to Batch-of-One.")
        is_single_run = True
        parameter_grid = [{"_job_type": "single_run"}]  # 伪造一个参数列表

    print(f"[Executor] Submitting Job Array with {len(parameter_grid)} tasks.")

    # 执行任务
    raw_result = _execute_batch_slurm(script_path, parameter_grid, timeout)

    # === 4. 结果解包修正 (Fix Return Type Mismatch) ===
    if is_single_run and raw_result["success"]:
        # 如果原本是单任务，metrics 当前是 List[Dict]，需要变成 Dict
        metrics_list = raw_result["metrics"]
        if isinstance(metrics_list, list) and len(metrics_list) > 0:
            raw_result["metrics"] = metrics_list[0]
        else:
            raw_result["metrics"] = {}  # 防御性编程

    return raw_result


def _execute_batch_slurm(script_path, parameter_grid, timeout):
    """
    内部辅助函数：处理 Slurm 提交、轮询和结果收集
    """
    working_dir = os.path.dirname(script_path)

    # 1. 提交作业
    submit_msg = submit_slurm_job_core(script_path, parameter_grid=parameter_grid)

    if "Success" not in submit_msg:
        return {
            "success": False,
            "error_message": submit_msg,
            "metrics": {},  # 注意这里为了兼容，如果是Batch失败也返回空Dict，由上层处理
            "output_files": [],
        }

    # 提取 Job ID
    job_id = submit_msg.split("ID: ")[-1].strip().rstrip(".")
    print(f"[Executor] Job {job_id} submitted. Waiting for completion...")

    # 2. 轮询等待
    start_time = time.time()
    while True:
        status = check_job_status_core(job_id)
        if status == "COMPLETED":
            break
        if time.time() - start_time > timeout:
            return {"success": False, "error_message": "Slurm execution timed out."}
        time.sleep(10)

    # 3. 收集结果
    # 无论是单次还是批量，只要是 Batch 模式提交的，文件名都带 ID (results_0.json)
    # 我们的 Programmer Prompt 保证了这一点
    json_files = glob.glob(os.path.join(working_dir, "results_*.json"))

    # 按索引排序 (results_0, results_1 ...)
    try:
        json_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    except Exception:
        pass

    if not json_files:
        err_files = glob.glob(os.path.join(working_dir, "*.err"))
        err_msg = "No result files found."
        if err_files:
            # 读取最新的 error log
            latest_err = max(err_files, key=os.path.getmtime)
            with open(latest_err, "r") as f:
                err_msg += (
                    f"\nLast Error Log ({os.path.basename(latest_err)}):\n{f.read()}"
                )

        return {
            "success": False,
            "error_message": err_msg,
            "metrics": {},
            "output_files": glob.glob(os.path.join(working_dir, "*")),
        }

    # 4. 读取数据
    results = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                results.append(json.load(f))
        except json.JSONDecodeError:
            print(f"[Executor] Warning: Failed to decode {jf}")

    return {
        "success": True,
        "output_summary": f"Collected {len(results)} results.",
        "metrics": results,
        "error_message": None,
        "output_files": json_files,
    }
