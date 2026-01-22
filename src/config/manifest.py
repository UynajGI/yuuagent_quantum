# src/config/manifest.py
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

# --- 子模块定义 ---


class TaskMeta(BaseModel):
    task_name: str
    description: str
    created_by: str = "Unknown"


class PhysicalModel(BaseModel):
    model_name: str = Field(description="TenPy 模型类名，如 RydbergChain")
    lattice_size: int = Field(gt=0, description="格点数 L")
    boundary_condition: Literal["open", "periodic", "infinite"]
    coupling_parameters: Dict[str, float] = Field(
        description="哈密顿量常数，如 Omega, V"
    )


class NumericalMethod(BaseModel):
    algorithm: Literal["dmrg", "idmrg", "tebd"]
    chi_max: int = Field(ge=10, description="最大截断维度")
    max_sweeps: int = 10
    convergence_threshold: float = 1e-7


class ExplorationPlan(BaseModel):
    scan_parameter: str = Field(description="扫描变量，如 delta")
    range_start: float
    range_end: float
    steps: int = Field(gt=1)
    adaptive_sampling: bool = False

    @validator("steps")
    def check_steps(cls, v):
        if v > 100:
            raise ValueError("Too many steps for a single batch!")
        return v


class ScientificGoals(BaseModel):
    observables: List[str]
    success_criteria: List[str]


# --- 根对象 ---


class ResearchManifest(BaseModel):
    """科研任务书总表"""

    task_meta: TaskMeta
    physical_model: PhysicalModel
    numerical_method: NumericalMethod
    exploration_plan: ExplorationPlan
    scientific_goals: ScientificGoals

    def to_prompt_context(self) -> str:
        """将结构化数据转换为 LLM 易读的 Prompt 摘要"""
        return f"""
Research Task: {self.task_meta.task_name}
Goal: {self.task_meta.description}

[Model Configuration]
- System: {self.physical_model.model_name} (L={self.physical_model.lattice_size})
- BC: {self.physical_model.boundary_condition}
- Parameters: {self.physical_model.coupling_parameters}

[Numerical Settings]
- Algo: {self.numerical_method.algorithm}
- Bond Dim (chi): {self.numerical_method.chi_max}

[Execution Plan]
- Scan '{self.exploration_plan.scan_parameter}': {self.exploration_plan.range_start} -> {self.exploration_plan.range_end} ({self.exploration_plan.steps} points)
- Adaptive: {self.exploration_plan.adaptive_sampling}

[Success Criteria]
- Track: {", ".join(self.scientific_goals.observables)}
"""
