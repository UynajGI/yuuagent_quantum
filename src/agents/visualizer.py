# src/agents/visualizer.py

import json
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

logger = logging.getLogger("Visualizer")


# 1. 决策结构
class VisualizationPlan(BaseModel):
    plot_type: Literal["line", "scatter", "heatmap", "bar", "none"] = Field(
        description="图表类型，如果不适合绘图则填 none"
    )
    x_var: Optional[str] = Field(description="X轴变量名")
    y_var: Optional[str] = Field(description="Y轴变量名")
    title: str = Field(description="图表标题")
    xlabel: str = Field(description="X轴标签")
    ylabel: str = Field(description="Y轴标签")
    save_filename: str = Field(description="保存文件名，如 result.png")


# 2. 初始化 LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 3. Prompt
viz_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个科学可视化专家。根据提供的数据样本，制定绘图计划。
规则：
1. 优先寻找控制参数（如 g, h, J, time）作为 X 轴。
2. 优先寻找序参量或能量（如 energy, mz, entropy）作为 Y 轴。
3. 如果数据是扫描过程（Scan），必须选择 'line' 或 'scatter'。
4. 确保物理单位正确。
""",
        ),
        (
            "human",
            """任务: {task_description}
可用列名: {available_columns}
数据样本 (前5行):
{data_sample}

请输出绘图计划:
{format_instructions}""",
        ),
    ]
)

parser = JsonOutputParser(pydantic_object=VisualizationPlan)
chain = viz_prompt | llm | parser


def _flatten_data(data: Union[List, Dict]) -> pd.DataFrame:
    """
    [关键修复] 智能扁平化数据，处理嵌套结构
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except:
            return pd.DataFrame()

    # Case 1: 已经是扁平的列表
    if isinstance(data, list):
        # 检查是否是 [{'metrics': {...}, 'job_id': 1}, ...] 这种结构
        if len(data) > 0 and isinstance(data[0], dict) and "metrics" in data[0]:
            # 提取 metrics 并合并外层字段 (如 job_id)
            flattened = []
            for item in data:
                row = item.get("metrics", {}).copy()
                # 把非 metrics 的其他简单字段也加进来 (如 params)
                for k, v in item.items():
                    if k != "metrics" and isinstance(v, (int, float, str)):
                        row[k] = v
                    elif k == "params" and isinstance(v, dict):
                        row.update(v)  # 展开 params
                flattened.append(row)
            return pd.DataFrame(flattened)
        else:
            return pd.DataFrame(data)

    # Case 2: 字典 (可能是 Aggregator 的输出)
    elif isinstance(data, dict):
        if "metrics" in data and isinstance(data["metrics"], list):
            return _flatten_data(data["metrics"])  # 递归处理
        else:
            return pd.DataFrame([data])

    return pd.DataFrame()


def create_visualization(
    task_description: str,
    aggregated_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    output_dir: str = ".",
) -> Dict[str, Any]:
    """
    生成科学图表
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 解析并清洗数据
    try:
        df = _flatten_data(aggregated_data)

        # 过滤掉非数值列，防止绘图报错
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if df.empty or not numeric_cols:
            return {
                "success": False,
                "error": "数据为空或没有数值列，无法绘图。",
                "save_path": None,
            }

    except Exception as e:
        return {"success": False, "error": f"数据解析失败: {e}", "save_path": None}

    # 2. 制定计划
    try:
        data_sample = df.head(5).to_markdown(index=False)
        plan = chain.invoke(
            {
                "task_description": task_description,
                "available_columns": ", ".join(numeric_cols),
                "data_sample": data_sample,
                "format_instructions": parser.get_format_instructions(),
            }
        )
    except Exception as e:
        return {"success": False, "error": f"LLM 绘图规划失败: {e}", "save_path": None}

    if plan["plot_type"] == "none":
        return {"success": True, "message": "LLM 决定不绘图。", "save_path": None}

    # 3. 执行绘图
    try:
        data_sample = df.head(5).to_markdown(index=False)

        # 获取原始 LLM 响应
        response_msg = chain.invoke(
            {
                "task_description": task_description,
                "available_columns": ", ".join(numeric_cols),
                "data_sample": data_sample,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        # 手动提取 JSON (DeepSeek 有时会多嘴)
        content = response_msg.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        # 解析 JSON
        plan_dict = json.loads(content)
        # 验证结构
        plan = VisualizationPlan(**plan_dict).dict()

    except Exception as e:
        return {
            "success": False,
            "error": f"LLM 绘图规划解析失败: {e}\nRaw Content: {content[:100]}...",
            "save_path": None,
        }
