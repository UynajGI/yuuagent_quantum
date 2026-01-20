# src/agents/conductor.py

import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

from src.agents.aggregator import aggregate_simulation_results
from src.agents.executor import execute_simulation_code
from src.agents.guide import guide_next_step
from src.agents.programmer import generate_tenpy_code

# === 引入所有 Agent ===
# 假设这些函数的签名已根据隔离原则进行了微调，只接收必要参数
from src.agents.strategist import decompose_task
from src.agents.validator import validate_simulation_results
from src.agents.visualizer import create_visualization

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Conductor")


# 1. 定义 Conductor 的决策结构
class ConductorDecision(BaseModel):
    next_action: Literal[
        "call_strategist",
        "call_guide",
        "call_programmer",
        "call_executor",
        "call_aggregator",
        "call_validator",
        "call_visualizer",
        "terminate",
    ] = Field(description="下一步调用的 Agent")

    context_for_agent: str = Field(description="传给 Agent 的上下文指令（隔离原则）")

    execution_params: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="若调用 Executor 进行批量扫描，在此提供参数列表（如 [{'h':0}, {'h':0.1}]）",
    )

    reasoning: str = Field(description="基于当前状态的决策理由")


# 2. 初始化 LLM
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

# 3. Conductor Prompt (体现论文图 1 的调度逻辑)
conductor_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是一个量子模拟自主实验的指挥官 (Conductor)。
你负责管理 Context Quarantine (上下文隔离) 并调度专门的 Agent。

### 标准工作流 (Workflow):
1. **Planning**: 如果没有计划，调用 `call_strategist`。
2. **Navigation**: 总是咨询 `call_guide` 来决定阶段（是继续跑模拟，还是分析，还是结束）。
3. **Implementation**:
   - 新任务 -> `call_programmer` (提供参数建议)。
   - 执行报错 -> `call_programmer` (提供 Traceback)。
   - 验证失败 -> `call_programmer` (提供 Validator 的物理反馈)。
4. **Execution**: 代码就绪 -> `call_executor`。
5. **Analysis Pipeline**:
   - 数据生成后 -> `call_aggregator` (清洗/汇总)。
   - 汇总后 -> `call_validator` (检查收敛性/物理合理性)。
   - 验证通过 -> `call_visualizer` (绘图)。

### 决策规则：
- **隔离原则**: 在 `context_for_agent` 字段中，只包含该 Agent **当前任务**所需的信息。不要复制整个历史。
- **错误处理**: 如果 `last_error` 存在，优先调用 `call_programmer` 进行修复。
- **验证优先**: 在绘图之前，必须经过 Validator 确认 `is_valid=True`。

当前系统状态：
- 已执行步骤: {executed_steps}
- 当前计划: {plan_status}
- 代码状态: {code_status} (Has Code: {has_code})
- 数据状态: {data_status} (Has Data: {has_data}, Validated: {is_validated})
- **最近错误**: {last_error}
""",
        ),
        ("human", "用户任务: {user_task}\n\n请决策: {format_instructions}"),
    ]
)

parser = JsonOutputParser(pydantic_object=ConductorDecision)
chain = conductor_prompt | llm | parser


def run_conductor(user_task: str, max_steps: int = 20):
    """
    执行符合 Context Quarantine 的自主科研循环
    """
    # === 状态存储 (State Management) ===
    state = {
        "plan": None,  # Strategist 输出
        "planning_history": [],  # Guide/Strategist 的专用对话历史
        "code": None,  # 当前 Python 脚本
        "raw_metrics": [],  # Executor 的原始输出列表
        "aggregated_data": None,  # Aggregator 的输出
        "last_error": None,  # 报错信息（Executor 或 Validator 产生）
        "is_validated": False,  # 是否通过物理验证
        "history_actions": [],  # 仅记录动作名，用于 Conductor 宏观判断
        "last_hypothesis": None,  # Guide 最近的科学假设
        "research_log": [],
        "repair_attempts": 0,
    }

    logger.info(f"🚀 Starting Mission: {user_task}")

    for step in range(max_steps):
        print(f"\n======== Step {step + 1} ========")

        if state["repair_attempts"] >= 3:
            logger.error("🚨 Critical: Repair limit reached. Infinite loop detected.")
            # 可以选择 terminate 或 request_human_help
            break

        # 准备状态描述供 Conductor 决策
        plan_status = "No Plan" if not state["plan"] else "Plan Active"
        code_status = "Ready" if state["code"] else "Missing"
        data_status = f"Raw: {len(state['raw_metrics'])} runs"

        # 1. Conductor 决策
        try:
            decision = chain.invoke(
                {
                    "user_task": user_task,
                    "executed_steps": state["history_actions"][-5:],  # 只看最近 5 步
                    "plan_status": plan_status,
                    "code_status": code_status,
                    "has_code": bool(state["code"]),
                    "data_status": data_status,
                    "has_data": bool(state["aggregated_data"]),
                    "is_validated": state["is_validated"],
                    "last_error": state["last_error"] or "None",
                    "repair_attempts": state["repair_attempts"],
                    "format_instructions": parser.get_format_instructions(),
                }
            )
        except Exception as e:
            logger.error(f"Conductor Brain Freeze: {e}")
            break

        action = decision["next_action"]
        context_input = decision["context_for_agent"]
        exec_params = decision.get("execution_params", None)
        reason = decision["reasoning"]

        log_entry = f"Step {step + 1}: Action={action} | Logic={reason}"
        state["research_log"].append(log_entry)

        logger.info(f"🤖 Decision: {action}")
        logger.info(f"📝 Logic: {reason}")

        if state["last_error"] and action == "call_programmer":
            state["repair_attempts"] += 1
            logger.info(f"🔧 Attempting Repair #{state['repair_attempts']}")
        elif not state["last_error"]:
            state["repair_attempts"] = 0

        # 2. 执行调度 (Context Quarantine Implementation)
        try:
            if action == "terminate":
                logger.info("✅ Workflow Terminated by Agent.")
                break

            # --- Planning Track ---
            elif action == "call_strategist":
                # Strategist 仅接收用户任务，不接收之前的报错干扰
                plan, state["planning_history"] = decompose_task(
                    user_task, state["planning_history"]
                )
                state["plan"] = plan
                logger.info(f"📋 Plan Updated: {len(plan.get('subtasks', []))} steps.")

            elif action == "call_guide":
                data_summary = (
                    str(state["aggregated_data"])
                    if state["aggregated_data"]
                    else "No data yet"
                )

                # 获取 Guide 的决策和更新的历史
                guide_decision, state["planning_history"] = guide_next_step(
                    user_task,
                    data_summary,
                    state["planning_history"],
                    current_plan=state["plan"],
                    validator_feedback=state["last_error"]
                    if state["is_validated"] is False
                    else None,
                    research_log=state["research_log"],
                )

                # === 关键保存：将假设存入全局状态 ===
                state["last_hypothesis"] = guide_decision.get("scientific_hypothesis")
                logger.info(f"🧪 New Hypothesis: {state['last_hypothesis']}")

                # 如果 Guide 建议调整参数，直接更新给 Conductor 决策参考
                if guide_decision.get("suggested_parameters"):
                    logger.info(
                        f"💡 Guide suggests params: {guide_decision['suggested_parameters']}"
                    )

            # --- Implementation Track ---
            elif action == "call_programmer":
                # 构建增强上下文：将 Guide 的假设注入给程序员
                enhanced_context = context_input
                if state.get("last_hypothesis"):
                    enhanced_context += f"\n\n[Scientific Hypothesis to Verify]:\n{state['last_hypothesis']}"

                # 如果有具体的参数建议，也一并传入
                # 这样程序员在写 argparse 的 default 值或者参数扫描范围时会有依据
                code_result = generate_tenpy_code(
                    task_description=user_task, context=enhanced_context
                )
                state["code"] = code_result["code"]

            # --- Execution Track ---
            elif action == "call_executor":
                if not state["code"]:
                    raise ValueError("No code to execute!")

                # Executor 运行
                final_params = exec_params if exec_params else None
                logger.info(
                    f"⚡ calling executor with {len(final_params) if final_params else 0} params"
                )
                exec_result = execute_simulation_code(
                    state["code"],
                    user_task,
                    parameter_grid=final_params,  # <--- 关键连接点
                )

                if exec_result["success"]:
                    # 成功：存入原始数据列表
                    metrics_data = exec_result["metrics"]
                    if isinstance(metrics_data, list):
                        state["raw_metrics"].extend(metrics_data)
                    else:
                        state["raw_metrics"].append(metrics_data)
                    state["last_error"] = None
                    logger.info("⚡ Execution Successful.")
                else:
                    # 失败：记录错误，下一轮 Conductor 会看到这个 error 并路由给 Programmer
                    state["last_error"] = exec_result["error_message"]
                    logger.warning(
                        f"💥 Execution Failed: {exec_result['error_message'][:100]}..."
                    )

            # --- Analysis Track ---
            elif action == "call_aggregator":
                # Aggregator 只负责清洗数据，不负责判断对错
                if not state["raw_metrics"]:
                    logger.warning("No data to aggregate.")
                    continue

                agg_result = aggregate_simulation_results(
                    user_task, state["raw_metrics"]
                )
                state["aggregated_data"] = agg_result
                logger.info("📊 Data Aggregated.")

            elif action == "call_validator":
                # [Context Quarantine]
                # Validator 绝对不能看代码，只看数据，防止被代码逻辑误导 (Hallucination)
                if not state["aggregated_data"]:
                    raise ValueError("No aggregated data to validate!")

                val_report = validate_simulation_results(
                    user_task, state["aggregated_data"]
                )
                state["is_validated"] = val_report["is_valid"]

                if not val_report["is_valid"]:
                    # 验证失败，将 issues 放入 last_error，迫使 Conductor 在下一步修正
                    state["last_error"] = f"Physics Violation: {val_report['issues']}"
                    logger.warning(f"🚫 Validation Failed: {val_report['issues']}")
                else:
                    logger.info(
                        f"✅ Validation Passed (Confidence: {val_report.get('confidence')})"
                    )

            elif action == "call_visualizer":
                if not state["is_validated"]:
                    logger.warning("⚠️ Warning: Plotting unvalidated data.")

                viz_result = create_visualization(user_task, state["aggregated_data"])
                logger.info(f"🎨 Plot saved: {viz_result.get('save_path')}")

            # 记录动作
            state["history_actions"].append(action)

        except Exception as e:
            logger.error(f"❌ Action {action} crashed: {e}")
            state["last_error"] = f"System Error in {action}: {str(e)}"

    return state
