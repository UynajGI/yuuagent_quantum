import json  # <--- ✅ 补上了！
import os
import shutil
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm  # 如果没安装 tqdm，可以去掉相关代码或 pip install tqdm

# ================= 1. 环境配置 (离线模式) =================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

# ================= 2. 路径配置 =================
PROJECT_ROOT = Path("/share/home/jiangyuan/yuuagent_quantum")
# 输入：刚才生成的清洗后的 JSON
CORPUS_JSON_PATH = PROJECT_ROOT / "src" / "knowledge" / "tenpy_corpus_clean.json"
# 输出：向量数据库路径
CHROMA_PATH = PROJECT_ROOT / "src" / "knowledge" / "chroma_db"

# ================= 3. Embedding 初始化 =================
# 必须与 loader.py 中的模型一致
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2", local_files_only=True
)


def flatten_metadata(meta: dict) -> dict:
    """
    ChromaDB 的 metadata 值只能是 str, int, float, bool。
    不能存 list 或 dict。我们需要把清洗脚本生成的复杂 metadata 拍平。
    """
    clean_meta = {}
    for k, v in meta.items():
        if isinstance(v, (list, dict)):
            # 将列表/字典转为字符串存储
            clean_meta[k] = str(v)
        elif v is None:
            clean_meta[k] = ""
        else:
            clean_meta[k] = v
    return clean_meta


def main():
    print("🚀 Starting Database Build from Clean Corpus...")

    # 1. 检查输入文件
    if not CORPUS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"❌ Corpus not found at {CORPUS_JSON_PATH}. Run build_clean_corpus.py first!"
        )

    # 2. 清理旧数据库 (强制重建，保证干净)
    if CHROMA_PATH.exists():
        print(f"🗑️  Cleaning old DB at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    # 3. 初始化 Chroma
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.create_collection(
        name="tenpy_knowledge", embedding_function=emb_fn
    )

    # 4. 加载语料
    print(f"📖 Loading corpus from {CORPUS_JSON_PATH}...")
    with open(CORPUS_JSON_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    print(f"🔹 Found {len(corpus)} items. Inserting into Vector DB...")

    # 5. 批量插入 (Batch Insert) - 提高效率
    BATCH_SIZE = 200
    ids_batch = []
    docs_batch = []
    metas_batch = []

    for item in tqdm(corpus, desc="Indexing"):
        # 准备数据
        # 你的清洗脚本生成的字段: type, name, file, content, summary, metadata

        # 构造 ID (确保唯一)
        # 清洗脚本里的 name 已经是唯一的了 (如 tenpy.algorithms.dmrg.TwoSiteDMRGEngine.run)
        doc_id = item["name"]

        # 构造文档内容
        # 如果有 summary，可以把 summary 加到 content 前面加强语义，或者直接存 content
        # 这里直接存 content (源码/完整文档)
        document = item["content"]

        # 构造 Metadata
        # 融合顶层字段和内层 metadata
        meta = {
            "type": item["type"],
            "name": item["name"],
            "file": item["file"],
            "summary": item.get("summary", "")[:1000],  # 限制 summary 长度
            # 标记是否为核心概念 (用于后续 loader.py 逻辑)
            "is_core": "doc_intro" in item["name"]
            or "doc_workflow" in item["name"]
            or "class" in item["type"],
        }

        # 融合清洗脚本提取的额外 metadata (如 args, bases)
        if "metadata" in item:
            meta.update(flatten_metadata(item["metadata"]))

        # 加入批次
        ids_batch.append(doc_id)
        docs_batch.append(document)
        metas_batch.append(meta)

        # 达到批次大小，提交
        if len(ids_batch) >= BATCH_SIZE:
            collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
            ids_batch = []
            docs_batch = []
            metas_batch = []

    # 6. 处理剩余数据
    if ids_batch:
        collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)

    print(f"\n✅ Vector DB build complete! Saved to {CHROMA_PATH}")
    print(f"📊 Total items indexed: {collection.count()}")


if __name__ == "__main__":
    main()
