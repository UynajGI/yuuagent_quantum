from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions

KNOWLEDGE_ROOT = Path(__file__).parent.resolve()
CHROMA_PATH = KNOWLEDGE_ROOT / "chroma_db"


class AdvancedRetriever:
    def __init__(self):
        # 1. 初始化 Embedding 函数
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_collection(
            name="tenpy_knowledge", embedding_function=self.emb_fn
        )

    def _get_core_knowledge(self) -> List[str]:
        """
        [方案三：核心注入]
        强制检索标记为 'core' 的文档（如 Workflow 模板、Model 基类）
        """
        results = self.collection.get(where={"is_core": True}, include=["documents"])
        return results["documents"] if results["documents"] else []

    def get_context(self, query: str, max_tokens: int = 45000) -> str:
        """
        构建防碎片化的结构化上下文
        """
        # 1. 加载常驻核心知识 (解决“只见树木不见森林”)
        final_chunks = self._get_core_knowledge()
        current_tokens = sum([len(c) // 4 for c in final_chunks])  # 粗略估计

        # 2. 执行向量检索 (Top 10)
        search_results = self.collection.query(
            query_texts=[query], n_results=10, include=["documents", "metadatas"]
        )

        seen_ids = set()
        retrieved_docs = search_results["documents"][0]
        retrieved_metas = search_results["metadatas"][0]

        # 3. [方案一：父文档追溯逻辑]
        context_blocks = []
        for doc, meta in zip(retrieved_docs, retrieved_metas):
            chunk_id = meta["name"]
            if chunk_id in seen_ids:
                continue

            # 碎片化修复：如果检索到的是类的方法，自动补全所属类的定义
            if meta["category"] == "api" and "." in chunk_id:
                parent_class_id = ".".join(chunk_id.split(".")[:-1])
                if parent_class_id not in seen_ids:
                    # 尝试从库里拉取父类的整体定义（包含 __init__）
                    parent_doc = self.collection.get(ids=[parent_class_id])
                    if parent_doc["documents"]:
                        context_blocks.append(
                            f"### PARENT CLASS: {parent_class_id} ###\n{parent_doc['documents'][0]}"
                        )
                        seen_ids.add(parent_class_id)

            context_blocks.append(f"### DETAIL: {chunk_id} ###\n{doc}")
            seen_ids.add(chunk_id)

        # 4. 组装并控制 Token 长度
        for block in context_blocks:
            block_tokens = len(block) // 4
            if current_tokens + block_tokens > max_tokens:
                break
            final_chunks.append(block)
            current_tokens += block_tokens

        # 5. 格式化输出 (符合论文要求的 Curated 风格)
        context_str = "\n\n".join(final_chunks)

        return f"""你现在拥有 TeNPy 的结构化参考文档。请严格遵循 API 签名和工作流建议。

{context_str}

[注意]：优先使用上述文档中的类和参数，不要编造不存在的属性。"""


# 单例模式
_retriever = AdvancedRetriever()


def get_tenpy_context(task_description: str) -> str:
    return _retriever.get_context(task_description)
