# src/tools/slurm.py
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.config.slurm_template import (
    DEFAULT_SLURM_CONFIG,
    SLURM_BATCH_TEMPLATE,
    SLURM_SCRIPT_TEMPLATE,
)

# === 1. 核心逻辑函数 (纯 Python，供 Executor 内部调用) ===


def submit_slurm_job_core(
    python_script_path: str,
    job_name: str = "tenpy_sim",
    parameter_grid: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Core logic for submitting Slurm jobs.
    """
    script_dir = os.path.dirname(python_script_path)
    script_name = os.path.basename(python_script_path)
    sbatch_path = os.path.join(script_dir, "run.sh")

    # 逻辑分支：批量 vs 单次
    if parameter_grid and len(parameter_grid) > 0:
        params_path = os.path.join(script_dir, "params.json")
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(parameter_grid, f, indent=2)

        sbatch_content = SLURM_BATCH_TEMPLATE.format(
            job_name=f"{job_name}_batch",
            log_dir=script_dir,
            cpus=DEFAULT_SLURM_CONFIG["cpus"],
            time_limit=DEFAULT_SLURM_CONFIG["time_limit"],
            partition=DEFAULT_SLURM_CONFIG["partition"],
            working_dir=script_dir,
            script_name=script_name,
            array_limit=len(parameter_grid) - 1,
        )
    else:
        log_path = os.path.join(script_dir, "slurm_output.log")
        sbatch_content = SLURM_SCRIPT_TEMPLATE.format(
            job_name=job_name,
            log_path=log_path,
            cpus=DEFAULT_SLURM_CONFIG["cpus"],
            time_limit=DEFAULT_SLURM_CONFIG["time_limit"],
            partition=DEFAULT_SLURM_CONFIG["partition"],
            working_dir=script_dir,
            script_name=script_name,
        )

    with open(sbatch_path, "w", encoding="utf-8") as f:
        f.write(sbatch_content)

    try:
        result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
        if result.returncode != 0:
            return f"Submission Failed: {result.stderr}"

        output = result.stdout.strip()
        job_id = output.split()[-1]

        mode = "Batch Job Array" if parameter_grid else "Single Job"
        return f"Success! {mode} submitted. ID: {job_id}"

    except FileNotFoundError:
        return "Error: 'sbatch' command not found."
    except Exception as e:
        return f"System Error: {str(e)}"


def check_job_status_core(job_id: str) -> str:
    """Core logic for checking status."""
    try:
        result = subprocess.run(
            ["squeue", "-j", job_id, "--noheader"], capture_output=True, text=True
        )
        if not result.stdout.strip():
            return "COMPLETED"
        else:
            return "RUNNING"
    except Exception:
        return "UNKNOWN"


# === 2. LangChain 工具定义 (供 LLM Agent 调用) ===


@tool
def submit_slurm_job(
    python_script_path: str,
    job_name: str = "tenpy_sim",
    parameter_grid: Optional[List[Dict[str, Any]]] = None,
):
    """
    Submit a Python script to the Slurm cluster.
    """
    # 简单地调用核心逻辑
    return submit_slurm_job_core(python_script_path, job_name, parameter_grid)


@tool
def check_job_status(job_id: str):
    """Check if a Slurm job is still running."""
    return check_job_status_core(job_id)
