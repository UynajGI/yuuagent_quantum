import ast
import os
import shutil
from pathlib import Path

import chromadb
import tiktoken
from chromadb.utils import embedding_functions

# 强制开启离线模式并清理代理
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# ================= 1. 路径与配置 =================
TENPY_ROOT = Path("/share/home/jiangyuan/yuuagent_quantum/tenpy")
CHROMA_PATH = Path(__file__).parent / "chroma_db"
EXAMPLES_SRC = TENPY_ROOT / "examples"
TENPY_PKG = TENPY_ROOT / "tenpy"
DOC_SRC = TENPY_ROOT / "doc"

# 初始化 Embedding
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", local_files_only=True
)

# Tiktoken 缓存配置
cache_dir = os.path.expanduser("~/.cache/tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = cache_dir
enc = tiktoken.get_encoding("cl100k_base")

# ================= 2. 核心存储工具 =================


def add_to_vector_db(
    collection, category: str, name: str, content: str, is_core: bool = False
):
    """将知识块存入 Chroma，传入 collection 对象"""
    if not content.strip():
        return

    tokens = len(enc.encode(content, disallowed_special=()))

    collection.add(
        ids=[name],
        documents=[content],
        metadatas=[
            {
                "category": category,
                "name": name,
                "tokens": tokens,
                "is_core": is_core,
            }
        ],
    )


# ================= 3. API 解析：原子化类 + 冗余方法 =================


class APIChunkVisitor(ast.NodeVisitor):
    def __init__(self, module_name: str, source_code: str, collection):
        self.module_name = module_name
        self.source_code = source_code
        self.collection = collection

    def visit_ClassDef(self, node):
        if node.name.startswith("_"):
            return

        class_content = ast.get_source_segment(self.source_code, node)
        class_id = f"{self.module_name}.{node.name}"

        is_core_class = any(
            kw in node.name for kw in ["Model", "DMRG", "MPS", "MPO", "Algorithm"]
        )
        add_to_vector_db(
            self.collection, "api", class_id, class_content, is_core=is_core_class
        )

        for item in node.body:
            if isinstance(item, ast.FunctionDef) and not item.name.startswith("_"):
                method_content = ast.get_source_segment(self.source_code, item)
                method_id = f"{class_id}.{item.name}"
                add_to_vector_db(
                    self.collection, "api", method_id, method_content, is_core=False
                )

    def visit_FunctionDef(self, node):
        if node.name.startswith("_"):
            return
        func_content = ast.get_source_segment(self.source_code, node)
        func_id = f"{self.module_name}.{node.name}"
        add_to_vector_db(self.collection, "api", func_id, func_content)


def process_api(collection):
    print("\n🔹 Processing API (Atomic Classes & Redundant Methods)...")
    for root, _, files in os.walk(TENPY_PKG):
        rel_path = Path(root).relative_to(TENPY_PKG)
        if "test" in str(rel_path) or "__" in str(rel_path):
            continue

        for pf in [f for f in files if f.endswith(".py")]:
            module_name = "tenpy." + ".".join(list(rel_path.parts) + [pf[:-3]]).replace(
                "..", "."
            )
            with open(Path(root) / pf, "r", encoding="utf-8") as f:
                code = f.read()
                visitor = APIChunkVisitor(module_name, code, collection)
                try:
                    visitor.visit(ast.parse(code))
                except Exception as e:
                    print(f"Skipping {module_name} due to parse error: {e}")


# ================= 4. 其他清洗逻辑 =================


def process_tutorials(collection):
    print("\n🔹 Processing Tutorials...")
    for rst in DOC_SRC.rglob("*.rst"):
        if any(bad in rst.name for bad in ["changelog", "release", "history"]):
            continue
        try:
            with open(rst, "r", encoding="utf-8") as f:
                content = f.read()
                is_core = "intro" in rst.name or "workflow" in rst.name
                add_to_vector_db(
                    collection, "tutorials", f"doc.{rst.stem}", content, is_core=is_core
                )
        except Exception as e:
            print(f"Error processing tutorial {rst.name}: {e}")


def process_examples(collection):
    print("\n🔹 Processing Examples...")
    for ex in EXAMPLES_SRC.rglob("*.py"):
        if "test" in ex.name:
            continue
        try:
            with open(ex, "r", encoding="utf-8") as f:
                add_to_vector_db(collection, "examples", f"example.{ex.stem}", f.read())
        except Exception as e:
            print(f"Error processing example {ex.name}: {e}")


# ================= 主程序 =================


def main():
    print("🚀 Starting Structured Knowledge Build")

    # 1. 彻底清理旧数据（避免 NFS 文件锁导致的元数据不同步）
    if CHROMA_PATH.exists():
        print(f"Cleaning old DB at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 2. 重新初始化 Client 和 Collection
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.create_collection(
        name="tenpy_knowledge", embedding_function=emb_fn
    )

    # 3. 运行处理函数
    process_api(collection)
    process_examples(collection)
    process_tutorials(collection)

    print(f"\n✅ Vector DB build complete at {CHROMA_PATH}")


if __name__ == "__main__":
    main()
