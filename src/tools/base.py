# src/tools/base.py
import os
import subprocess

from langchain_core.tools import tool


# 1. 文件写入工具
@tool
def write_file(file_path: str, content: str):
    """Write content to a file. Useful for saving Python scripts or configuration."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


# 2. 文件读取工具
@tool
def read_file(file_path: str):
    """Read the content of a file. Useful for inspecting logs or data."""
    try:
        if not os.path.exists(file_path):
            return "File does not exist."
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


# 3. 代码执行工具
@tool
def run_python_script(script_path: str, timeout: int = 300):
    """Execute a Python script and return stdout/stderr.
    Use this to run simulations or plotting scripts.
    """
    try:
        result = subprocess.run(
            ["python", script_path], capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return f"Execution Success:\n{result.stdout}"
        else:
            return (
                f"Execution Failed:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
            )
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out."
    except Exception as e:
        return f"Error executing script: {str(e)}"


# 导出工具列表供 Agent 绑定
ALL_TOOLS = [write_file, read_file, run_python_script]
