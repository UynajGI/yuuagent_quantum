# src/config/slurm_template.py

# 这是一个 Jinja2 风格或者是 f-string 风格的模板
# 请根据你所在集群（login01）的实际情况修改分区名、模块加载等
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}

# === 环境配置 (根据你的集群修改) ===
source ~/.bashrc

# === 运行目录 ===
cd {working_dir}

# === 执行命令 ===
echo "Starting job at $(date)"
python {script_name}
echo "Job finished at $(date)"
"""

# 默认配置（防止 Agent 乱填）
DEFAULT_SLURM_CONFIG = {
    "partition": "cpu_amd",  # 改为你集群的分区名，如 'cpu', 'short', 'compute'
    "cpus": 4,
    "time_limit": "02:00:00",
}
