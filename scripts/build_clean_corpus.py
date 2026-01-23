# scripts/build_clean_corpus.py
import ast
import json
import os
import re

# ================= 配置路径 =================
PROJECT_ROOT = "/share/home/jiangyuan/yuuagent_quantum"
PATHS = {
    "source_code": os.path.join(PROJECT_ROOT, "tenpy", "tenpy"),
    "examples": os.path.join(PROJECT_ROOT, "tenpy", "examples"),
    "docs": os.path.join(PROJECT_ROOT, "tenpy", "doc"),
}
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "src", "knowledge", "tenpy_corpus_clean.json")


# ================= 1. 源代码解析器 (修复版：支持类作用域) =================
class CodeVisitor(ast.NodeVisitor):
    def __init__(self, filename, relative_path):
        self.filename = filename
        self.relative_path = relative_path
        self.chunks = []
        # 计算模块名: tenpy/tenpy/linalg/charges.py -> tenpy.tenpy.linalg.charges
        self.module_name = relative_path.replace("/", ".").replace(".py", "")
        # === 关键修复：类作用域栈 ===
        self.class_stack = []

    def visit_ClassDef(self, node):
        """提取类定义"""
        docstring = ast.get_docstring(node) or ""
        source_segment = ast.get_source_segment(self.source_code, node)

        # 类名 ID: module.ClassName
        full_class_name = f"{self.module_name}.{node.name}"

        self.chunks.append(
            {
                "type": "api_class",
                "name": full_class_name,
                "file": self.relative_path,
                "content": source_segment,
                "summary": docstring.split("\n\n")[0]
                if docstring
                else "No description.",
                "metadata": {
                    "bases": [b.id for b in node.bases if isinstance(b, ast.Name)],
                    "methods": [
                        n.name for n in node.body if isinstance(n, ast.FunctionDef)
                    ],
                },
            }
        )

        # === 关键修复：入栈 -> 访问子节点 -> 出栈 ===
        self.class_stack.append(node.name)
        self.generic_visit(node)  # 继续遍历类内部的方法
        self.class_stack.pop()

    def visit_FunctionDef(self, node):
        """提取函数定义 (包括类方法和独立函数)"""
        # 忽略私有方法，但保留 __init__ 和 __call__
        if node.name.startswith("_") and node.name not in ["__init__", "__call__"]:
            return

        docstring = ast.get_docstring(node) or ""
        source_segment = ast.get_source_segment(self.source_code, node)

        # === 关键修复：根据栈判断是方法还是函数 ===
        if self.class_stack:
            # 这是一个类方法: module.ClassName.method_name
            parent_class = self.class_stack[-1]
            unique_name = f"{self.module_name}.{parent_class}.{node.name}"
            func_type = "api_method"
        else:
            # 这是一个顶层函数: module.function_name
            unique_name = f"{self.module_name}.{node.name}"
            func_type = "api_function"

        self.chunks.append(
            {
                "type": func_type,
                "name": unique_name,
                "file": self.relative_path,
                "content": source_segment,
                "summary": docstring.split("\n\n")[0] if docstring else "",
                "metadata": {
                    "args": [a.arg for a in node.args.args],
                    "parent_class": self.class_stack[-1] if self.class_stack else None,
                },
            }
        )
        # 函数内部定义的函数一般不需要提取，不再递归

    def parse(self):
        with open(self.filename, "r", encoding="utf-8") as f:
            self.source_code = f.read()

        try:
            tree = ast.parse(self.source_code)
            self.visit(tree)
        except Exception as e:
            print(f"⚠️ AST Parse Error in {self.filename}: {e}")
        return self.chunks


# ================= 2. 文档清洗器 (保持不变) =================
def clean_rst(text: str) -> str:
    text = re.sub(r"\.\. \w+::.*", "", text)
    text = re.sub(r":\w+ .*?:", "", text)
    text = re.sub(r":\w+:`(.*?)`", r"\1", text)
    text = re.sub(r"`(.*?)\s<.*?>`_", r"\1", text)
    lines = [line.strip() for line in text.split("\n")]
    clean_lines = [l for l in lines if l]
    return "\n".join(clean_lines)


def process_docs(doc_root):
    chunks = []
    for root, _, files in os.walk(doc_root):
        for file in files:
            if not file.endswith(".rst"):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, doc_root)

            with open(full_path, "r", encoding="utf-8") as f:
                raw_content = f.read()

            clean_content = clean_rst(raw_content)

            if len(clean_content) > 100:
                chunks.append(
                    {
                        "type": "doc_tutorial",
                        "name": f"doc_{rel_path.replace('/', '_').replace('.rst', '')}",
                        "file": rel_path,
                        "content": clean_content,
                        "summary": clean_content[:200],
                        "metadata": {"format": "rst_cleaned"},
                    }
                )
    return chunks


# ================= 3. 示例处理 (保持不变) =================
def process_examples(example_root):
    chunks = []
    for root, _, files in os.walk(example_root):
        for file in files:
            if not file.endswith(".py"):
                continue

            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, example_root)

            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks.append(
                {
                    "type": "example_script",
                    "name": f"example_{file}",
                    "file": rel_path,
                    "content": f"### Example Script: {file} ###\n{content}",
                    "summary": f"Full executable example: {file}",
                    "metadata": {"executable": True},
                }
            )
    return chunks


# ================= 主流程 =================
def main():
    all_knowledge = []
    seen_ids = set()

    print("🚀 Starting Knowledge ETL (Scope-Aware Version)...")

    # 1. Process Source Code
    print("Parsing Source Code (AST)...")
    for root, _, files in os.walk(PATHS["source_code"]):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                visitor = CodeVisitor(full_path, rel_path)
                chunks = visitor.parse()
                all_knowledge.extend(chunks)

    # 2. Process Docs
    print("Cleaning Documentation (RST)...")
    all_knowledge.extend(process_docs(PATHS["docs"]))

    # 3. Process Examples
    print("Loading Examples...")
    all_knowledge.extend(process_examples(PATHS["examples"]))

    # === 4. 去重处理 (Final Deduplication) ===
    unique_knowledge = []
    for item in all_knowledge:
        if item["name"] in seen_ids:
            # 如果真的还有重复 (比如 if/else 定义了两次同名函数)，跳过或加后缀
            continue
        seen_ids.add(item["name"])
        unique_knowledge.append(item)

    print(f"✅ Extracted {len(unique_knowledge)} unique knowledge chunks.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(unique_knowledge, f, indent=2, ensure_ascii=False)

    print(f"💾 Corpus saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
