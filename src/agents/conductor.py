# src/agents/conductor.py

import logging
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from src.agents.aggregator import aggregate_simulation_results
from src.agents.executor import execute_simulation_code
from src.agents.programmer import generate_tenpy_code

# === 引入所有 Agent ===
from src.agents.strategist import decompose_task
from src.agents.validator import validate_simulation_results
from src.agents.visualizer import create_visualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Conductor")


# 定义决策结构
class ConductorDecision(BaseModel):
    next_action: Literal[
        "call_strategist",
        "call_programmer",
        "call_executor",
        "call_aggregator",
        "call_validator",
        "call_visualizer",
        "terminate",
    ] = Field(description="下一步调用的 Agent")
    reasoning: str = Field(description="决策理由")


llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

conductor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子模拟自主实验的指挥官 (Conductor)。
你需要根据当前状态，按照科学研究的逻辑调度子 Agent。

标准工作流 (TeNPy Simulation Workflow):
1. **Strategist**: 用户刚提出请求，需要制定计划。
2. **Programmer**: 有了计划或需要修改代码时，编写/修复代码。
3. **Executor**: 有了代码，运行模拟。
4. **Validator**: 模拟完成，检查结果是否收敛/物理合理。
   - 如果 Validator 报错 -> 回退给 Programmer (带上错误信息)。
   - 如果 Validator 通过 -> 进入下一步。
5. **Aggregator/Guide**: 分析数据，决定是否需要更多参数扫描。
6. **Visualizer**: 所有数据准备好后，绘图。
7. **Terminate**: 得到最终图表或多次失败后终止。

当前状态：
- 历史动作: {executed_steps}
- 上一步输出: {last_output_summary}
- 错误信息: {last_error}
""",
        ),
        ("human", "用户任务: {user_task}\n\n请决策: {format_instructions}"),
    ]
)

parser = JsonOutputParser(pydantic_object=ConductorDecision)
chain = conductor_prompt | llm | parser


def run_conductor(user_task: str, max_steps: int = 15):
    """
    执行自主科研循环
    """
    state = {
        "user_task": user_task,
        "history": [],
        "last_output": None,
        "last_error": None,
        "code": None,  # 暂存生成的代码
        "data": None,  # 暂存模拟数据
        "plan": None,  # 暂存研究计划
    }

    for step in range(max_steps):
        print(f"\n======== Step {step + 1} ========")

        # 1. Conductor 决策
        decision = chain.invoke(
            {
                "user_task": user_task,
                "executed_steps": [h["action"] for h in state["history"]],
                "last_output_summary": str(state["last_output"])[
                    :500
                ],  # 截断防止 Token 溢出
                "last_error": state["last_error"],
                "format_instructions": parser.get_format_instructions(),
            }
        )

        action = decision["next_action"]
        reason = decision["reasoning"]
        logger.info(f"🤖 Conductor Decision: {action} ({reason})")

        current_output = {}

        # 2. 执行调度
        try:
            if action == "terminate":
                logger.info("✅ Mission Completed or Aborted.")
                break

            elif action == "call_strategist":
                output = decompose_task(user_task)
                state["plan"] = output
                current_output = f"Plan created: {output.get('subtasks')}"

            elif action == "call_programmer":
                # 将计划或之前的错误传给程序员
                context = (
                    state["last_error"] if state["last_error"] else str(state["plan"])
                )
                output = generate_tenpy_code(user_task, context=context)
                state["code"] = output["code"]
                current_output = "Code generated."

            elif action == "call_executor":
                if not state["code"]:
                    raise ValueError("No code to execute!")
                # 执行代码
                output = execute_simulation_code(state["code"], user_task)
                if not output["success"]:
                    state["last_error"] = output["error_message"]
                else:
                    state["data"] = output["metrics"]  # 假设 metrics 是结果字典
                    state["last_error"] = None
                current_output = f"Execution done. Success: {output['success']}"

            elif action == "call_validator":
                if not state["data"]:
                    raise ValueError("No data to validate!")
                output = validate_simulation_results(user_task, state["data"])
                if not output["is_valid"]:
                    state["last_error"] = f"Validation Failed: {output['issues']}"
                current_output = f"Validation: {output['is_valid']}"

            elif action == "call_aggregator":
                # 在论文中 Aggregator 负责汇总多次运行，这里简化为单次
                output = aggregate_simulation_results(user_task, state["data"])
                current_output = output["summary"]

            elif action == "call_visualizer":
                output = create_visualization(user_task, state["data"])
                current_output = f"Plot saved to {output.get('save_path')}"

            # 更新状态
            state["last_output"] = current_output
            state["history"].append({"action": action, "output": current_output})

        except Exception as e:
            logger.error(f"❌ Action failed: {e}")
            state["last_error"] = str(e)

    return state
