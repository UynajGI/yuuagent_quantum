# src/knowledge/loader.py
from pathlib import Path

# 定位知识库根目录
KNOWLEDGE_ROOT = Path(__file__).parent.resolve()


def _read_file(rel_path: str) -> str:
    """读取知识文件内容的辅助函数"""
    path = KNOWLEDGE_ROOT / rel_path
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def get_tenpy_context(task_description: str) -> str:
    """
    根据任务描述动态组装 Context。

    参数:
        task_description: 用户的任务描述 (e.g., "Simulate 2D Ising model using DMRG")

    返回:
        str: 拼装好的 Prompt 上下文 (包含 API, Examples, Tutorials)
    """
    task_lower = task_description.lower()
    context_parts = []

    # 1. === CORE API (基础必读) ===
    # Agent 必须知道如何定义模型和网格
    context_parts.append("### TeNPy CORE API ###")
    context_parts.append(_read_file("api/models.txt"))
    context_parts.append(_read_file("api/networks.txt"))

    # 2. === ALGORITHMS (按需加载) ===
    # 总是加载算法基础，因为通常都需要运行某种模拟
    context_parts.append(_read_file("api/algorithms.txt"))

    # 如果涉及复杂模拟配置 (Simulations class)
    if "simulation" in task_lower or "resume" in task_lower:
        context_parts.append(_read_file("api/simulations.txt"))

    # 如果涉及线性代数/张量操作 (Linalg)
    if any(k in task_lower for k in ["contract", "tensor", "svd", "npc"]):
        context_parts.append(_read_file("api/linalg.txt"))
        context_parts.append(_read_file("api/tools.txt"))

    # 3. === EXAMPLES (模仿学习) ===
    context_parts.append("\n### REFERENCE CODE PATTERNS ###")

    example_dir = KNOWLEDGE_ROOT / "examples"
    all_examples = list(example_dir.glob("*.txt"))
    selected_examples = []

    # 策略：根据关键词匹配最相关的示例
    if "time" in task_lower or "evolution" in task_lower or "tdvp" in task_lower:
        # 时间演化任务
        selected_examples.extend(
            [
                f
                for f in all_examples
                if "time" in f.name or "tebd" in f.name or "tdvp" in f.name
            ]
        )
    elif "infinite" in task_lower or "imps" in task_lower:
        # 无限长系统
        selected_examples.extend([f for f in all_examples if "infinite" in f.name])
    elif "custom" in task_lower:
        # 自定义模型
        selected_examples.extend(
            [f for f in all_examples if "custom" in f.name or "model" in f.name]
        )

    # 如果没匹配到特定类型，默认加载基础 DMRG
    if not selected_examples:
        selected_examples.extend(
            [f for f in all_examples if "dmrg" in f.name and "finite" in f.name]
        )

    # 限制数量，防止 Token 爆炸 (取前 2 个)
    for ex in selected_examples[:2]:
        content = _read_file(f"examples/{ex.name}")
        if content:
            context_parts.append(f"\n# Example: {ex.stem}\n{content}")

    # 4. === TUTORIALS (概念补充) ===
    context_parts.append("\n### CONCEPTUAL GUIDE ###")
    context_parts.append(_read_file("tutorials/intro.txt"))

    if "model" in task_lower:
        context_parts.append(_read_file("tutorials/models.txt"))

    return "\n\n".join(context_parts)
