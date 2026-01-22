# src/agents/validator.py

import json
from typing import Any, Dict, List, Union

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


# 1. Output Structure
class ValidationReport(BaseModel):
    is_valid: bool = Field(
        description="Is the result physically reasonable and numerically reliable?"
    )
    issues: List[str] = Field(
        description="List of issues found (e.g., 'Energy not converged', 'Mz > 0.5')"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Validation confidence (1.0 = fully confident)"
    )
    recommendations: List[str] = Field(
        description="Suggestions for improvement (e.g., 'Increase bond dimension')"
    )


# 2. Initialize LLM
# [Improvement]: Use deepseek-chat for standard JSON output stability,
# or ensure deepseek-reasoner output is stripped of <think> tags if used.
# Here we stick to 'deepseek-chat' with low temperature for reliable parsing.
llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_retries=2,
)

# 3. Prompt Construction
validator_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a Quantum Many-Body Simulation Validator Agent.
Your task is to STRICTLY verify simulation results against physical laws and numerical consistency.

### Physics Knowledge Base (Rydberg & Ising Focus):
1. **Transverse Field Ising**:
   - Order parameter <sigma_z> should decrease as transverse field 'g' increases.
   - Ground state energy must be monotonic with 'g'.
   - At critical point (g~1), correlation length should peak.
2. **Rydberg Atom Chain**:
   - Z2 Phase: Atoms alternate (up-down-up-down).
   - Paramagnetic Phase: Uniform state.
3. **DMRG/MPS Checks**:
   - **Energy Variance**: Should be < 1e-5 for reliable results.
   - **Entanglement Entropy**: Should not be negative.
   - **Truncation Error**: High truncation error (> 1e-4) implies insufficient bond dimension (chi).

### Protocol:
- If `energy` increases during a cooling/ground-state search, report INVALID.
- If `magnetization` behaves opposite to physical intuition, report INVALID.
- If data is empty or NaNs are present, report INVALID.

Input Context:
Task: {task_description}
""",
        ),
        (
            "human",
            """Simulation Results Summary:
{simulation_results}

Analyze and report in JSON format:
{format_instructions}""",
        ),
    ]
)

parser = JsonOutputParser(pydantic_object=ValidationReport)
chain = validator_prompt | llm | parser


def validate_simulation_results(
    task_description: str,
    simulation_results: Union[str, Dict[str, Any], List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Validate simulation results for physical plausibility.
    """
    if isinstance(simulation_results, (dict, list)):
        input_str = json.dumps(simulation_results, indent=2, ensure_ascii=False)
    else:
        input_str = str(simulation_results)

    # [Improvement]: Check for empty data before calling LLM
    if not input_str or input_str == "[]" or input_str == "{}":
        return {
            "is_valid": False,
            "issues": ["No data generated to validate."],
            "confidence": 1.0,
            "recommendations": ["Check Executor or Programmer for runtime errors."],
        }

    try:
        # [Improvement]: Truncate extremely long data to prevent context overflow
        # but keep the head and tail which often contain the most relevant trend info
        if len(input_str) > 10000:
            input_segment = input_str[:5000] + "\n...[snipped]...\n" + input_str[-5000:]
        else:
            input_segment = input_str

        result = chain.invoke(
            {
                "task_description": task_description,
                "simulation_results": input_segment,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        return result
    except Exception as e:
        return {
            "is_valid": False,
            "issues": [f"Validation crashed: {str(e)}"],
            "confidence": 0.0,
            "recommendations": ["Manual inspection required."],
        }
