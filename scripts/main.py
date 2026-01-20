# main.py
import os
import sys

from dotenv import load_dotenv

# 1. 加载环境变量 (必须在 import langchain 之前)
load_dotenv()

# 确保能找到 src 包
sys.path.append(r"/share/home/jiangyuan/yuuagent_quantum")

# 1. 强制清理代理 (防止计算节点连接本地代理失败)
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["ALL_PROXY"] = ""

# 2. 强制开启离线模式 (防止 HuggingFace 尝试联网)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


from src.agents.conductor import run_conductor  # noqa: E402
from src.build_knowledge import main as build_knowledge


def run_test():
    print("🔒 正在检查环境配置...")
    if not os.getenv("LANGSMITH_API_KEY"):
        print("❌ 错误：未找到 LANGSMITH_API_KEY")
        return

    # 2. (可选) 首次运行前重建知识库，确保 ChromaDB 存在
    # 如果你之前跑过 build_knowledge.py，这步可以注释掉
    if not os.path.exists("src/knowledge/chroma_db"):
        build_knowledge()
        print("📚 正在初始化知识库 (RAG)...")

    # 3. 定义一个简单的测试任务 (由简入繁)
    # 使用一个计算量小、容易验证的任务，例如 Ising 模型的小尺寸计算
    test_task = (
        "Run a DMRG simulation for a 1D Transverse Field Ising Model. "
        "Parameters: L=10, J=1.0, g=1.5. "
        "Calculate the ground state energy and average magnetization Mz. "
        "No need to plot, just output the values."
    )

    print(f"🚀 启动任务: {test_task}")
    print(
        f"📡 LangSmith Tracing: {'ENABLED' if os.getenv('LANGSMITH_TRACING') == 'true' else 'DISABLED'}"
    )
    print(f"📊 Project: {os.getenv('LANGSMITH_PROJECT')}")
    print("-" * 50)

    # 4. 运行指挥官 (限制步数防止死循环)
    final_state = run_conductor(user_task=test_task, max_steps=10)

    print("-" * 50)
    print("✅ 任务结束")
    print(f"最终状态摘要: {len(final_state['research_log'])} steps executed.")
    if final_state["last_error"]:
        print(f"⚠️ 最终报错: {final_state['last_error']}")
    else:
        print("🎉 似乎成功了！")


if __name__ == "__main__":
    run_test()
