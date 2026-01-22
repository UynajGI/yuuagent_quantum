import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ================= 配置 =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# 你的语料库路径
CORPUS_PATH = PROJECT_ROOT / "src" / "knowledge" / "tenpy_corpus_clean.json"


class SymbolicLoader:
    def __init__(self):
        print(f"📚 [SymbolicLoader] Loading cleaned corpus from: {CORPUS_PATH}")
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(f"❌ Corpus not found: {CORPUS_PATH}")

        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        # === 索引构建 ===
        self.full_name_index: Dict[str, dict] = {}
        self.short_name_index: Dict[str, List[dict]] = {}

        # 🌟 新增：专门存放示例脚本的列表
        self.example_library: List[dict] = []

        # 核心文档
        self.core_docs: List[str] = []

        self._build_index()
        print(
            f"✅ [SymbolicLoader] Indexed {len(self.corpus)} items. Example Library size: {len(self.example_library)}"
        )

    def _build_index(self):
        for item in self.corpus:
            name = item["name"]
            item_type = item.get("type", "")

            # 1. 专门归档示例脚本
            if item_type == "example_script" or "examples" in item["file"]:
                self.example_library.append(item)

            # 2. 归档核心文档
            elif "doc_intro" in name or "doc_workflow" in name:
                self.core_docs.append(item["content"])

            # 3. 通用索引 (全名 & 短名)
            self.full_name_index[name] = item

            short_name = name.split(".")[-1]
            if short_name not in self.short_name_index:
                self.short_name_index[short_name] = []
            self.short_name_index[short_name].append(item)

    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词，用于匹配"""
        # 提取字母数字组合，转小写
        words = set(re.findall(r"[a-zA-Z_0-9]+", text.lower()))
        # 过滤停用词
        stopwords = {
            "main",
            "print",
            "len",
            "simulation",
            "python",
            "calculate",
            "using",
            "for",
            "the",
            "and",
            "model",
        }
        return {w for w in words if w not in stopwords and len(w) > 2}

    def _find_best_examples(self, task_description: str, limit: int = 2) -> List[str]:
        """
        🌟 核心逻辑：根据任务描述，找到最匹配的示例脚本
        """
        keywords = self._extract_keywords(task_description)
        scored_examples: List[Tuple[int, dict]] = []

        for ex in self.example_library:
            score = 0
            ex_name = ex["name"].lower()
            ex_content = ex["content"].lower()

            # 简单评分机制
            for kw in keywords:
                # 文件名包含关键词 (权重高)
                if kw in ex_name:
                    score += 10
                # 内容包含关键词 (权重低)
                elif kw in ex_content:
                    score += 1

            if score > 0:
                scored_examples.append((score, ex))

        # 按分数降序排列
        scored_examples.sort(key=lambda x: x[0], reverse=True)

        # 返回前 N 个
        results = []
        for score, ex in scored_examples[:limit]:
            print(f"   💡 Found relevant example: {ex['name']} (Score: {score})")
            results.append(
                f"### 🔥 REFERENCE EXAMPLE: {ex['name']} ###\n{ex['content']}"
            )

        return results

    def get_context(self, task_description: str, error_context: str = "") -> str:
        """
        智能上下文组装
        """
        final_context = []

        # === 场景 A: 报错修复 (Debug Mode) ===
        # 优先级最高：如果报错了，必须查源码
        if error_context and ("Traceback" in error_context or "Error" in error_context):
            print("🕵️ [SymbolicLoader] Debug Mode Activated.")
            keywords = self._extract_keywords(error_context)

            for kw in keywords:
                # 在短名索引里找 (比如 'run')
                if kw in self.short_name_index:
                    hits = self.short_name_index[kw]
                    # 优先找 API 定义，排除 example (debug 时不需要 example，要看底层实现)
                    hits = [h for h in hits if h["type"] != "example_script"]
                    # 排序：优先 tenpy 库文件
                    hits.sort(
                        key=lambda x: 1 if "tenpy" in x["name"] else 0, reverse=True
                    )

                    for hit in hits[:2]:
                        final_context.append(
                            f"### CRITICAL SOURCE CODE: {hit['name']} ###\n{hit['content']}"
                        )

            return "\n\n".join(final_context)

        # === 场景 B: 正常编程 (Exploration Mode) ===
        # 优先级：示例脚本 > 核心文档 > API定义
        print("📖 [SymbolicLoader] Exploration Mode (Example-First Strategy).")

        # 1. 🔥 注入最匹配的示例脚本 (这是你最想要的！)
        best_examples = self._find_best_examples(task_description)
        final_context.extend(best_examples)

        # 2. 注入核心文档 (Intro/Workflow) 用于补充概念
        if not best_examples:  # 如果没找到示例，多放点文档
            for doc in self.core_docs[:3]:
                final_context.append(f"### CORE DOC ###\n{doc[:2000]}...")
        else:
            # 如果有示例，文档少放点，省 token
            for doc in self.core_docs[:1]:
                final_context.append(f"### CORE DOC ###\n{doc[:1000]}...")

        # 3. 补充一些 API 定义 (基于关键词)
        # 比如提到了 TFIModel，就把 TFIModel 的类定义放进去
        keywords = self._extract_keywords(task_description)
        for kw in keywords:
            if kw in self.short_name_index:
                hits = self.short_name_index[kw]
                # 只看类定义，且不是 example
                hits = [
                    h
                    for h in hits
                    if h["type"] == "api_class" and "example" not in h["name"]
                ]
                for hit in hits[:1]:
                    final_context.append(
                        f"### API REFERENCE: {hit['name']} ###\n{hit['summary']}"
                    )

        return "\n\n".join(final_context)

    def lookup_specific(self, name: str) -> str:
        if name in self.full_name_index:
            return self.full_name_index[name]["content"]
        return ""


# === 单例 ===
try:
    _loader = SymbolicLoader()
except Exception as e:
    print(f"⚠️ SymbolicLoader init failed: {e}")
    _loader = None


# === 导出接口 ===
def get_tenpy_context(task_description: str, context: str = "") -> str:
    if _loader:
        err_ctx = context if context and "Traceback" in context else ""
        return _loader.get_context(task_description, error_context=err_ctx)
    return ""


def lookup_specific_api(symbol_name: str) -> str:
    if _loader:
        return _loader.lookup_specific(symbol_name)
    return ""
