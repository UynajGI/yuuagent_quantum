import ast
import os
import re
from pathlib import Path

import tiktoken  # <--- 新增导入

# ================= 1. 路径配置 =================
# TeNPy 源码根目录 (根据您的环境配置)
TENPY_ROOT = Path("/share/home/jiangyuan/yuuagent_quantum/tenpy")

# 输出目录 (当前脚本所在目录下的 knowledge 文件夹)
SRC_KNOWLEDGE = Path(__file__).parent / "knowledge"

# 源目录映射
EXAMPLES_SRC = TENPY_ROOT / "examples"
TENPY_PKG = TENPY_ROOT / "tenpy"
DOC_SRC = TENPY_ROOT / "doc"

# 目标子目录
DIRS = {
    "examples": SRC_KNOWLEDGE / "examples",
    "api": SRC_KNOWLEDGE / "api",
    "tutorials": SRC_KNOWLEDGE / "tutorials",
}

# ================= 2. 基础工具 =================


def ensure_env():
    """初始化目录结构"""
    if SRC_KNOWLEDGE.exists():
        # 可选：清理旧数据，保证纯净
        # shutil.rmtree(SRC_KNOWLEDGE)
        pass

    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    (SRC_KNOWLEDGE / "__init__.py").touch()
    print(f"📂 Knowledge Base initialized at: {SRC_KNOWLEDGE}")


def write_file(path: Path, content: str):
    """写入文件辅助函数"""
    if not content.strip():
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    # 打印相对路径，保持日志整洁
    print(f"  - Generated: {path.relative_to(SRC_KNOWLEDGE)}")


# ================= 3. 清洗逻辑：Examples =================


def clean_example_code(file_path: Path) -> str:
    """清洗示例代码：移除绘图、非必要打印，保留物理逻辑"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return ""  # 跳过二进制或非文本文件

    cleaned_lines = [f"# Source: {file_path.name}"]

    for line in lines:
        # 1. 移除绘图相关 (Visualizer Agent 的工作)
        if any(
            kw in line
            for kw in [
                "matplotlib",
                "pyplot",
                "plt.",
                "seaborn",
                ".show()",
                ".savefig",
                "fig, ax",
            ]
        ):
            continue

        # 2. 移除冗长的打印 (如版本号、欢迎语)
        stripped = line.strip()
        if stripped.startswith("print(") and (
            "-" * 5 in line or "version" in line or "TeNPy" in line
        ):
            continue

        cleaned_lines.append(line.rstrip())

    return "\n".join(cleaned_lines)


def process_examples():
    print("\n🔹 Processing Examples...")
    # 递归查找所有 .py 文件
    all_examples = list(EXAMPLES_SRC.rglob("*.py"))

    for ex in all_examples:
        # 跳过测试和配置文件
        if "test" in ex.name or "conftest" in ex.name:
            continue

        content = clean_example_code(ex)
        if len(content) < 50:
            continue  # 忽略太短的文件

        # 命名策略：扁平化处理，防止同名冲突
        # 例如: userguide/d_dmrg.py -> userguide_d_dmrg.txt
        if ex.parent != EXAMPLES_SRC:
            safe_name = f"{ex.parent.name}_{ex.name}"
        else:
            safe_name = ex.name

        target = DIRS["examples"] / safe_name.replace(".py", ".txt")
        write_file(target, content)


# ================= 4. 清洗逻辑：API (AST解析 - 增强版) =================


class APIVisitor(ast.NodeVisitor):
    """AST 访问者：提取类、带有默认值的签名和文档摘要"""

    def __init__(self):
        self.output = []
        self.current_class = None

    def _format_arg(self, arg):
        """辅助函数：处理带类型注解的参数"""
        if arg.annotation:
            try:
                # Python 3.9+ 支持 ast.unparse
                ann = ast.unparse(arg.annotation)
                return f"{arg.arg}: {ann}"
            except AttributeError:
                pass
        return arg.arg

    def _get_args_str(self, args_node):
        """辅助函数：重建带有默认值的参数列表"""
        args = []
        defaults = args_node.defaults
        n_args = len(args_node.args)
        n_defaults = len(defaults)

        # 处理位置参数
        for i, arg in enumerate(args_node.args):
            arg_str = self._format_arg(arg)
            # 检查是否有默认值
            default_idx = i - (n_args - n_defaults)
            if default_idx >= 0:
                try:
                    default_val = ast.unparse(defaults[default_idx])
                    arg_str += f"={default_val}"
                except AttributeError:
                    arg_str += "=..."  # Fallback for complex defaults
            args.append(arg_str)

        # 处理关键字参数 (kwonlyargs)
        for i, arg in enumerate(args_node.kwonlyargs):
            arg_str = self._format_arg(arg)
            if i < len(args_node.kw_defaults) and args_node.kw_defaults[i] is not None:
                try:
                    default_val = ast.unparse(args_node.kw_defaults[i])
                    arg_str += f"={default_val}"
                except AttributeError:
                    arg_str += "=..."
            args.append(arg_str)

        if args_node.vararg:
            args.append(f"*{args_node.vararg.arg}")
        if args_node.kwarg:
            args.append(f"**{args_node.kwarg.arg}")

        return ", ".join(args)

    def visit_ClassDef(self, node):
        if node.name.startswith("_"):
            return  # 跳过私有类
        self.current_class = node.name

        doc = ast.get_docstring(node)
        doc_sum = doc.split("\n")[0] if doc else ""

        self.output.append(f"\nclass {node.name}:")
        if doc_sum:
            self.output.append(f'    """{doc_sum}"""')

        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        # 跳过私有方法/函数
        if self.current_class:
            if node.name.startswith("_") and node.name != "__init__":
                return
        else:
            if node.name.startswith("_"):
                return

        # 1. 提取完整的函数签名 (带默认值)
        arg_str = self._get_args_str(node.args)

        # 2. 提取文档摘要
        doc = ast.get_docstring(node)
        doc_sum = doc.split("\n")[0] if doc else ""

        # 3. 格式化输出
        indent = "    " if self.current_class else ""
        if not doc_sum:
            self.output.append(f"{indent}def {node.name}({arg_str}): pass")
        else:
            self.output.append(f"{indent}def {node.name}({arg_str}):")
            self.output.append(f'{indent}    """{doc_sum}"""')


def extract_api_from_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return ""

    visitor = APIVisitor()
    visitor.visit(tree)
    return "\n".join(visitor.output)


def process_api():
    print("\n🔹 Processing API (Source Code)...")
    # 遍历 tenpy 包下所有子目录
    for root, _, files in os.walk(TENPY_PKG):
        rel_path = Path(root).relative_to(TENPY_PKG)

        # 跳过测试目录和缓存目录
        if "test" in str(rel_path) or "__" in str(rel_path):
            continue

        py_files = [f for f in files if f.endswith(".py") and not f.startswith("test")]
        if not py_files:
            continue

        module_content = []
        for pf in py_files:
            content = extract_api_from_file(Path(root) / pf)
            if content.strip():
                module_content.append(f"# Module: tenpy.{rel_path}.{pf[:-3]}")
                module_content.append(content)

        if module_content:
            # 生成文件名：将路径斜杠转换为下划线
            safe_name = str(rel_path).replace("/", "_")
            if safe_name == ".":
                safe_name = "core"

            write_file(DIRS["api"] / f"{safe_name}.txt", "\n\n".join(module_content))


# ================= 5. 清洗逻辑：Tutorials (RST) =================


def clean_rst_content(file_path: Path) -> str:
    """清洗 RST 文档：去除 Sphinx 指令，保留文本、代码块和公式"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return ""

    # 1. 简化引用链接 :class:`~tenpy.models.Model` -> Model
    text = re.sub(r":[a-zA-Z0-9_-]+:`~?([a-zA-Z0-9_.-]+)`", r"\1", text)

    lines = text.splitlines()
    cleaned = []

    for line in lines:
        stripped = line.strip()
        # 移除 Sphinx 指令行 (.. xxxx::) 但保留 math (如果 math 紧跟内容通常在下一行缩进)
        if stripped.startswith("..") and "::" in stripped:
            if "math::" in stripped:
                continue  # math 内容在下一行，保留
            continue  # 其他指令跳过

        # 移除引用定义和索引
        if stripped.startswith(".. _") or "toctree::" in stripped:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def process_tutorials():
    print("\n🔹 Processing Tutorials (Documentation)...")
    all_rst = list(DOC_SRC.rglob("*.rst"))

    # 1. 垃圾词黑名单 (文件名匹配)
    IGNORE_KEYWORDS = [
        "release",
        "history",
        "what",
        "news",
        "upgrade",
        "index",
        "bib",
        "ref",
        "literat",
        "paper",
        "author",
        "credit",
        "license",
        "copyright",
        "install",
        "contribut",
        "ack",
        "todo",
        "trouble",
        "faq",
        "main",
        "changelog",
        "pip",
        "conda",
        "from_source",
        "test",
        "updating",
        "extra",
        "base",
        "class",
        "module",
        "build_doc",
        "logging",
        "overview",
        "introductions",
        "guidelines",
    ]

    categories = {"intro": [], "models": [], "algorithms": [], "advanced": []}

    for rst in all_rst:
        lower_name = rst.name.lower()
        lower_path = str(rst).lower()

        # --- 过滤逻辑 ---

        # 1. 路径黑名单：如果文件在 changelog 文件夹里，直接扔掉
        if "changelog" in lower_path:
            # print(f"  [Skipped Path] {rst.relative_to(DOC_SRC)}")
            continue

        # 2. 文件名黑名单
        if any(bad in lower_name for bad in IGNORE_KEYWORDS):
            continue

        content = clean_rst_content(rst)
        if len(content) < 100:
            continue

        formatted = f"\n\n--- DOC: {rst.stem} ---\n{content}"

        # --- 分类逻辑 ---
        if "model" in lower_name:
            categories["models"].append(formatted)
        elif any(
            k in lower_name
            for k in ["algorithm", "dmrg", "tdvp", "tebd", "vumps", "contract"]
        ):
            categories["algorithms"].append(formatted)
        elif any(
            k in lower_name
            for k in [
                "guide",
                "intro",
                "lattice",
                "mps",
                "mpo",
                "site",
                "input",
                "output",
            ]
        ):
            categories["intro"].append(formatted)
        else:
            categories["advanced"].append(formatted)

    # 写入文件
    for cat, contents in categories.items():
        if contents:
            write_file(DIRS["tutorials"] / f"{cat}.txt", "".join(contents))


# ================= 6. Token 统计逻辑 (Tiktoken) =================


def estimate_token_usage():
    print("\n📊 Precise Token Usage (via tiktoken cl100k_base)")
    print("=" * 70)
    print(f"{'Category':<15} | {'File Name':<35} | {'Tokens':>10}")
    print("-" * 70)

    total_tokens = 0
    category_tokens = {}

    try:
        # DeepSeek-V3 兼容 cl100k_base 编码
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"⚠️  Error loading tiktoken: {e}")
        return

    for root, dirs, files in os.walk(SRC_KNOWLEDGE):
        for file in files:
            if not file.endswith(".txt"):
                continue

            path = Path(root) / file
            # 父文件夹名作为分类 (api, examples, tutorials)
            category = Path(root).name

            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                    # 核心统计：使用 tiktoken 编码
                    tokens = len(enc.encode(content, disallowed_special=()))

                    total_tokens += tokens
                    category_tokens[category] = (
                        category_tokens.get(category, 0) + tokens
                    )

                    # 打印单文件统计 (文件名截断)
                    display_name = (file[:32] + "..") if len(file) > 32 else file
                    print(f"{category:<15} | {display_name:<35} | {tokens:>10}")
            except Exception:
                pass

    print("=" * 70)
    print(f"📈 TOTAL EXACT TOKENS: {total_tokens}")
    print("   (DeepSeek Context Limit: 128k. Safe prompt size: < 100k)")
    print("-" * 70)
    for cat, count in category_tokens.items():
        print(f"   - {cat:<12}: {count} tokens")
    print("=" * 70)


# ================= 主程序 =================


def main():
    print("🚀 Starting TeNPy Knowledge Build")
    print(f"   Source Root: {TENPY_ROOT}")

    ensure_env()
    process_examples()
    process_api()
    process_tutorials()

    # 执行 Token 统计
    estimate_token_usage()

    print(f"\n✅ Build Complete! Knowledge base ready at: {SRC_KNOWLEDGE}")


if __name__ == "__main__":
    main()
