# src/agents/visualizer.py

import json
import os
from typing import Any, Dict, List, Literal, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. 定义 Visualizer 的决策输出结构
class VisualizationPlan(BaseModel):
    plot_type: Literal["line", "scatter", "heatmap", "phase_diagram", "bar"] = Field(
        description="图表类型"
    )
    x_var: str = Field(description="X 轴变量名（如 'h', 'time'）")
    y_var: str = Field(description="Y 轴变量名（如 'mz', 'energy'）")
    title: str = Field(description="图表标题")
    xlabel: str = Field(description="X 轴标签（含单位）")
    ylabel: str = Field(description="Y 轴标签（含单位）")
    save_path: str = Field(description="保存路径（如 'mz_vs_h.pdf'）")


# 2. 初始化 LLM（仅用于决策，temperature=0）
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)


# 3. Prompt：让 LLM 决定如何可视化
viz_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个科学可视化专家（Visualizer Agent）。你的任务是：
- 根据聚合后的数据和原始研究目标，决定最佳可视化方案
- 选择合适的图表类型（折线图用于扫描参数，热力图用于二维相图等）
- 提供清晰的坐标轴标签（含物理单位）
- 输出文件名应具有描述性（如 'ising_mz_vs_h.pdf'）

不要生成图像，只输出绘图指令。
""",
        ),
        (
            "human",
            """研究任务：{task_description}

聚合数据摘要：
{aggregated_data}

可用变量包括：{available_columns}

请按以下 JSON Schema 输出绘图计划：
{format_instructions}""",
        ),
    ]
)


# 4. 绑定解析器
parser = JsonOutputParser(pydantic_object=VisualizationPlan)
chain = viz_prompt | llm | parser


# 5. 核心函数：生成并保存图表
def create_visualization(
    task_description: str,
    aggregated_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    生成出版级科学图表

    Args:
        task_description: 用户原始任务
        aggregated_data: Aggregator 返回的结构化数据
        output_dir: 图表保存目录

    Returns:
        包含 save_path 和状态的字典
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 标准化数据为 DataFrame
    try:
        if isinstance(aggregated_data, str):
            data = json.loads(aggregated_data)
        else:
            data = aggregated_data

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported data format")

        available_cols = list(df.columns)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse data: {str(e)}",
            "save_path": None,
        }

    # 让 LLM 决定如何画
    try:
        plan = chain.invoke(
            {
                "task_description": task_description,
                "aggregated_data": str(data)[:2000],
                "available_columns": ", ".join(available_cols),
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Visualization planning failed: {str(e)}",
            "save_path": None,
        }

    # 实际绘图
    try:
        plt.figure(figsize=(8, 6))
        sns.set(style="whitegrid")

        x = df[plan["x_var"]]
        y = df[plan["y_var"]]

        if plan["plot_type"] == "line":
            plt.plot(x, y, marker="o")
        elif plan["plot_type"] == "scatter":
            plt.scatter(x, y)
        elif plan["plot_type"] == "bar":
            plt.bar(x, y)
        else:
            # 默认折线图
            plt.plot(x, y, marker="o")

        plt.title(plan["title"])
        plt.xlabel(plan["xlabel"])
        plt.ylabel(plan["ylabel"])
        plt.tight_layout()

        save_path = os.path.join(output_dir, plan["save_path"])
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {"success": True, "save_path": save_path, "plot_info": plan}

    except Exception as e:
        return {
            "success": False,
            "error": f"Plotting failed: {str(e)}",
            "save_path": None,
        }
