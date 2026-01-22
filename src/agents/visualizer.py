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


# 1. Decision Structure
class VisualizationPlan(BaseModel):
    plot_type: Literal["line", "scatter", "heatmap", "bar", "none"] = Field(
        description="Type of plot to generate. Use 'none' if data is unsuitable for plotting."
    )
    x_var: Optional[str] = Field(description="Column name for X axis")
    y_var: Optional[str] = Field(description="Column name for Y axis")
    title: str = Field(description="Scientific title for the plot")
    xlabel: str = Field(description="Label for X axis (with units)")
    ylabel: str = Field(description="Label for Y axis (with units)")
    save_filename: str = Field(description="Filename only (e.g., 'phase_diagram.png')")


# 2. Initialize LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 3. Prompt
viz_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Scientific Visualization Expert.
Analyze the provided data columns and task description to create a publication-quality plot plan.

Rules:
1. Identify the 'Control Parameter' (e.g., h, g, delta) for the X-axis.
2. Identify the 'Order Parameter' or 'Observable' (e.g., E, M, S_vN) for the Y-axis.
3. If multiple Y-values exist for one X (e.g., from different seeds), prefer a Scatter plot or Line plot with markers.
4. Ensure physical units are mentioned in labels if known (e.g., 'Time [1/J]').
""",
        ),
        (
            "human",
            """Task: {task_description}
Available Columns: {available_columns}
Data Sample (First 5 rows):
{data_sample}

Generate Plotting Plan:
{format_instructions}""",
        ),
    ]
)

parser = JsonOutputParser(pydantic_object=VisualizationPlan)
chain = viz_prompt | llm | parser


def create_visualization(
    task_description: str,
    aggregated_data: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    output_dir: str = ".",  # Passed dynamically from Conductor
) -> Dict[str, Any]:
    """
    Generate scientific plots from aggregated data.
    """
    # 1. Parse Data
    try:
        if isinstance(aggregated_data, str):
            data = json.loads(aggregated_data)
        else:
            data = aggregated_data

        # Normalize to List of Dicts
        if isinstance(data, dict):
            # Check if it's a "summary" wrapper
            if "metrics" in data and isinstance(data["metrics"], list):
                data = data["metrics"]
            else:
                data = [data]

        df = pd.DataFrame(data)

        # [Improvement]: Filter out non-numeric columns for plotting
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not numeric_cols:
            return {
                "success": False,
                "error": "No numeric data to plot.",
                "save_path": None,
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"Data parsing failed: {e}",
            "save_path": None,
        }

    # 2. Plan
    try:
        # Only send a sample to LLM to save tokens
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
        return {"success": False, "error": f"Planning failed: {e}", "save_path": None}

    if plan["plot_type"] == "none":
        return {
            "success": True,
            "message": "LLM decided not to plot.",
            "save_path": None,
        }

    # 3. Plot
    try:
        # [Improvement]: Use high-quality science style
        plt.style.use("seaborn-v0_8-paper")
        plt.figure(figsize=(10, 6))

        # Check if columns exist
        if plan["x_var"] not in df.columns or plan["y_var"] not in df.columns:
            raise ValueError(
                f"Selected columns {plan['x_var']}/{plan['y_var']} not in data."
            )

        # Sort by X for clean line plots
        df_sorted = df.sort_values(by=plan["x_var"])
        x = df_sorted[plan["x_var"]]
        y = df_sorted[plan["y_var"]]

        if plan["plot_type"] == "line":
            plt.plot(x, y, "o-", linewidth=2, markersize=6, label=plan["y_var"])
        elif plan["plot_type"] == "scatter":
            plt.scatter(x, y, alpha=0.8, label=plan["y_var"])
        elif plan["plot_type"] == "bar":
            plt.bar(x, y)

        plt.title(plan["title"], fontsize=14)
        plt.xlabel(plan["xlabel"], fontsize=12)
        plt.ylabel(plan["ylabel"], fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()

        # Save
        full_save_path = os.path.join(output_dir, plan["save_filename"])
        plt.savefig(full_save_path, dpi=300, bbox_inches="tight")
        plt.close()

        return {"success": True, "save_path": full_save_path, "plot_info": plan}

    except Exception as e:
        logger.error(f"Plotting routine crashed: {e}")
        return {"success": False, "error": f"Matplotlib error: {e}", "save_path": None}
