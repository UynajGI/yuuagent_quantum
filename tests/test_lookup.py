# test_lookup.py

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.knowledge.loader import get_tenpy_context, lookup_specific_api


def test_retrieval():
    print("🔍 测试 1: 模拟 Conductor 查词 (Active Lookup)")
    # 这就是 Conductor 在报错时会调用的函数
    source_code = lookup_specific_api("TwoSiteDMRGEngine.run")

    if "return E, psi" in source_code:
        print("✅ 成功！找到了关键的 return 语句！")
        print("-" * 20)
        print(source_code[:500] + "...\n(后面省略)")
    else:
        print("❌ 失败！没有找到 return E, psi。")
        print("搜索结果摘要:", source_code[:200])

    print("\n" + "=" * 50 + "\n")

    print("🧠 测试 2: 模拟智能检索 (Traceback Detection)")
    # 模拟一个报错信息
    error_query = """
    Traceback (most recent call last):
      File "simulation.py", line 58, in main
        E = engine.run()
    TypeError: cannot unpack non-iterable float object
    """
    # 看看 get_tenpy_context 能不能自动提取上下文
    context = get_tenpy_context(error_query, max_tokens=2000)
    print(f"检索到的上下文长度: {len(context)} chars")
    if "def run" in context:
        print("✅ 上下文里包含了 run 函数的定义！")
    else:
        print("⚠️ 上下文里没找到 run 函数，可能需要依赖 Conductor 的手动查词。")


if __name__ == "__main__":
    test_retrieval()
