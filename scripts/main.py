import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# --- 1. 环境初始化 (必须最先执行) ---
def setup_environment():
    """配置 HPC 运行环境：加载 .env，清理代理，强制离线"""
    load_dotenv()

    # 添加项目根目录到 sys.path
    project_root = r"/share/home/jiangyuan/yuuagent_quantum"
    if project_root not in sys.path:
        sys.path.append(project_root)

    # 强制清理代理 (防止连接计算节点失败)
    os.environ["HTTP_PROXY"] = ""
    os.environ["HTTPS_PROXY"] = ""
    os.environ["ALL_PROXY"] = ""

    # 强制开启离线模式
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


setup_environment()

# --- 2. 导入模块 (环境配置后导入) ---
# 确保你已经创建了 src/schema/manifest.py (参考上一个回答)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.agents.conductor import run_conductor
from src.build_knowledge import main as build_knowledge
from src.config.manifest import ResearchManifest

# --- 3. 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("YuuAgent_Main")

# --- 4. 辅助函数 ---


def check_knowledge_base():
    """检查并自动构建 ChromaDB"""
    db_path = Path("src/knowledge/chroma_db")
    if not db_path.exists():
        logger.warning("📚 知识库不存在，正在初始化 (RAG)...")
        try:
            build_knowledge()
            logger.info("✅ 知识库构建完成")
        except Exception as e:
            logger.error(f"❌ 知识库构建失败: {e}")
            sys.exit(1)
    else:
        logger.info("📚 知识库已就绪")


def load_manifest(json_path: str) -> ResearchManifest:
    """加载并验证任务书 (The Gatekeeper)"""
    path = Path(json_path)
    if not path.exists():
        logger.error(f"❌ 找不到任务文件: {json_path}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Pydantic 强类型验证
        manifest = ResearchManifest(**data)
        logger.info(f"✅ 任务书 '{manifest.task_meta.task_name}' 格式校验通过")
        return manifest

    except json.JSONDecodeError:
        logger.error(f"❌ JSON 语法错误: {json_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 任务书内容不合规:\n{e}")
        sys.exit(1)


# --- 5. 主入口 ---


def run():
    # A. 检查 LangSmith
    if not os.getenv("LANGSMITH_API_KEY"):
        logger.warning("⚠️ 未检测到 LANGSMITH_API_KEY，Tracing 可能无法使用")

    # B. 准备知识库
    check_knowledge_base()

    # C. 获取输入文件路径
    # 优先读取命令行参数，否则使用默认测试文件
    if len(sys.argv) > 1:
        task_file = sys.argv[1]
    else:
        task_file = "scripts/task.json"
        logger.info(f"ℹ️ 未指定输入文件，使用默认测试: {task_file}")

    # D. 加载并转换任务
    manifest = load_manifest(task_file)

    # 关键步骤：将结构化对象转换为 Conductor 能理解的 Prompt Context
    task_context_str = manifest.to_prompt_context()

    print("\n" + "=" * 50)
    print(f"🚀 启动 YuuAgent: {manifest.task_meta.task_name}")
    print(f"📋 任务摘要:\n{task_context_str.strip()}")
    print("=" * 50 + "\n")

    # E. 启动指挥官
    final_state = run_conductor(
        user_task=task_context_str,
        max_steps=20,  # 稍微放宽步数限制
    )

    # F. 结束报告
    print("\n" + "-" * 50)
    if final_state.get("last_error"):
        logger.error(f"⚠️ 任务异常终止: {final_state['last_error']}")
    else:
        logger.info(f"🎉 任务完成! 执行步数: {len(final_state['history_actions'])}")
        # 这里可以加代码打印最终结果文件的路径
        if final_state.get("aggregated_data"):
            print(f"📊 最终数据摘要: {str(final_state['aggregated_data'])[:200]}...")


if __name__ == "__main__":
    run()
