# YuuAgent-Quantum: Autonomous Quantum Simulation with LangChain Agents

YuuAgent-Quantum 是一个基于 LangChain 框架构建的自主量子模拟系统，旨在通过 AI 代理协作自动执行量子多体物理模拟任务。系统能够理解用户的物理问题，生成适当的模拟代码，执行计算，验证结果并生成可视化图表。

## 🧠 系统架构

本系统采用多智能体协作架构，包含以下核心组件：

### Agent 层
- **Strategist**: 任务分解专家，将用户需求分解为可执行的子任务
- **Programmer**: 代码生成专家，基于 TenPy 库生成量子模拟代码
- **Executor**: 执行器，运行生成的模拟代码并收集结果
- **Validator**: 验证器，检查结果的物理合理性和数值可靠性
- **Aggregator**: 数据聚合器，分析和总结模拟结果
- **Guide**: 科研向导，建议下一步行动策略
- **Visualizer**: 可视化专家，生成出版级科学图表
- **Conductor**: 指挥官，协调各代理按科学逻辑执行任务

### 知识层
- **Knowledge Loader**: 动态检索 TenPy 相关文档和示例
- **API 文档**: 提供 TenPy 库的实时参考
- **示例库**: 包含常用量子模型的实现示例

## 📋 目录结构

```
yuuagent_quantum/
├── src/
│   ├── agents/           # 智能体实现
│   │   ├── strategist.py    # 任务分解
│   │   ├── programmer.py    # 代码生成
│   │   ├── executor.py      # 代码执行
│   │   ├── validator.py     # 结果验证
│   │   ├── aggregator.py    # 数据聚合
│   │   ├── guide.py         # 决策引导
│   │   ├── visualizer.py    # 结果可视化
│   │   └── conductor.py     # 任务调度
│   ├── knowledge/        # 知识库组件
│   │   ├── loader.py       # 知识加载器
│   │   ├── api/           # API文档
│   │   ├── examples/      # 示例代码
│   │   └── tutorials/     # 教程资料
│   ├── config/          # 配置文件
│   ├── tools/           # 工具类
│   └── build_knowledge.py # 知识库构建脚本
├── scripts/             # 可执行脚本
├── input/               # 输入文件
├── output/              # 输出结果
├── assets/              # 资源文件
├── configs/             # 配置文件
├── docs/                # 文档
├── logs/                # 日志文件
├── notebooks/           # Jupyter 笔记本
├── README.md            # 项目说明
├── USAGE.md             # 使用指南
├── pyproject.toml       # Python 项目配置
└── yuuskel.toml         # 项目模板配置
```

## 🚀 快速开始

### 环境准备

1. 安装 Python 3.10+ 环境
2. 设置 DeepSeek API 密钥到 `.env` 文件
3. 安装依赖包

```bash
pip install uv  # 推荐使用 uv 作为包管理器
uv sync --all-extras  # 安装所有依赖
```

或者使用 pip：

```bash
pip install -e .
```

### 环境变量配置

项目根目录的 `.env` 文件应包含以下变量：

```bash
DEEPSEEK_API_KEY=your_api_key_here
OUTPUT_DIR=output
INPUT_DIR=input
ASSETS_DIR=assets
```

### 运行示例

```bash
# 加载环境变量
set -a; source .env; set +a

# 运行自主量子模拟
python -c "
from src.agents.conductor import run_conductor
result = run_conductor('Run DMRG simulation of transverse field Ising model with h=1.0')
print(result)
"
```

## 🔬 功能特性

- **自主任务分解**: 自动将复杂的物理问题分解为可执行步骤
- **代码生成与优化**: 基于 TenPy 库生成高质量量子模拟代码
- **智能验证**: 验证结果的物理合理性和数值收敛性
- **动态决策**: 根据结果调整后续计算策略
- **出版级可视化**: 自动生成高质量科学图表
- **可复现工作流**: 完整的日志记录和结果跟踪

## 🛠️ 技术栈

- **AI/ML Frameworks**: LangChain, DeepSeek API
- **Quantum Computing**: TenPy (Tensor Network Python)
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Development**: Pydantic, Ruff, PyTest

## 📚 使用示例

典型的使用场景包括：

1. **相变研究**: 自动扫描参数空间寻找相变点
2. **基态计算**: 使用 DMRG/Tebd 算法计算基态性质
3. **动力学模拟**: 执行时间演化计算
4. **模型比较**: 对比不同哈密顿量模型的结果

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进此项目！

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情