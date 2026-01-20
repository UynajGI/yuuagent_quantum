# src/config/slurm_template.py

# 单任务模板 (保持不变)
SLURM_SCRIPT_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={log_path}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}

source ~/.bashrc
cd {working_dir}

echo "Starting job at $(date)"
python {script_name}
echo "Job finished at $(date)"
"""

# === 新增：批量任务模板 (Job Array) ===
# 关键点：
# 1. --array=0-{array_limit} 开启数组作业
# 2. $SLURM_ARRAY_TASK_ID 获取当前任务索引
# 3. 传递 --param_file 和 --job_id 给 Python 脚本
SLURM_BATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/job_%a.out
#SBATCH --error={log_dir}/job_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --partition={partition}
#SBATCH --array=0-{array_limit}%{max_array_size}

source ~/.bashrc
cd {working_dir}

echo "Starting Batch Task $SLURM_ARRAY_TASK_ID at $(date)"
# 核心：将 Job ID 传给脚本，脚本读取 params.json 中的第 ID 项参数
python {script_name} --param_file params.json --job_id $SLURM_ARRAY_TASK_ID
echo "Batch Task $SLURM_ARRAY_TASK_ID finished at $(date)"
"""

DEFAULT_SLURM_CONFIG = {
    "partition": "cpu_amd",
    "cpus": 4,
    "time_limit": "04:00:00",
    "max_array_size": 100,
}
