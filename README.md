# CRH-AI

# CRH-AI：基于计算现实假说的几何相干底座

**下一代类生命智能架构 | 从符号大模型 → 几何锚定智能**

> 核心思想：为大模型植入**先天时空几何底座**，让 AI 不再悬浮于概率符号，而是扎根宇宙几何拓扑，从根源解决幻觉、因果缺失、直觉缺失问题。
> 理论基础：**CRH 计算现实假说（Computational Reality Hypothesis）**，统一几何、熵、纠缠、拓扑与认知。

---

## 🧠 架构总览（四层几何相干底座）

```
┌─────────────────────────────────────────────────────────────────┐
│  第 4 层：弱相干重构与融合层  (Weak Coherence Reconstruction)  │
│  - 残缺信息补全、多模态几何对齐、场景记忆关联                   │
└─────────────────────────────────────────────────────────────────┘
                    ↓↑ 相干流
┌─────────────────────────────────────────────────────────────────┐
│  第 3 层：物理先天约束层        (Physical Prior Constraints)    │
│  - 因果不可逆约束、空间不可穿透、能量守恒、拓扑稳定性          │
└─────────────────────────────────────────────────────────────────┘
                    ↓↑ 纠缠梯度
┌─────────────────────────────────────────────────────────────────┐
│  第 2 层：纠缠梯度感知层        (Entanglement Gradient Field)   │
│  - 局部强纠缠、全局弱场、注意力动态加权                          │
└─────────────────────────────────────────────────────────────────┘
                    ↓↑ 坐标流
┌─────────────────────────────────────────────────────────────────┐
│  第 1 层：时空几何锚定层        (Geometric Anchor Layer)        │
│  - MDS 多维尺度嵌入、Schwarzschild 曲率场、ρ 信息密度分布      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 三大增强功能（2026 年 5 月更新）

### A. 动态 ρ 更新机制（Online Learning Loop）
让 AI 通过学习不断进化自己的几何世界模型。

**核心特性**：
- 从验证事实中学习，更新几何流形
- 四种事实类型：physical（物理事实）、causal（因果事实）、mathematical（数学事实）、contradiction（矛盾事实）
- 自动更新背景曲率场，触发局部拓扑重构

**使用示例**：
```python
bridge.learn_from_feedback({
    'summary': '地球是近似球形',
    'type': 'physical',
    'region': 'solar_system',
    'confidence': 0.95
})
```

### B. 几何提示词增强（Geometric Prompt Engineering）
让 LLM 接收的不再是纯文本，而是带几何描述子的结构化提示。

**增强提示示例**：
```
[几何相干上下文]
- 对称性: Spherically Symmetric
- 曲率强度: Medium-High
- 纠缠梯度: Strong-Local
- 拓扑结构: Hopfion-like
- 因果方向: Forward
- 当前自洽性: 0.927

用户查询: 描述黑洞视界附近的光线行为

请在以上几何约束下进行推理...
```

### C. 多尺度 MDL 剪枝（Multiscale Pruning）
根据问题复杂度动态选择流形分辨率，实现计算效率与精度平衡。

**复杂度 → 尺度映射**：
| 复杂度分数 | 尺度 | 说明 |
|-----------|------|------|
| < 0.3 | 1 | 极低维粗粒化（最快） |
| 0.3 ~ 0.6 | 2 | 低维 |
| 0.6 ~ 0.85 | 4 | 中高维 |
| ≥ 0.85 | 8 | 高维精细流形 |

---

## 📦 安装

```bash
# 克隆项目
git clone https://github.com/yourusername/CRH-AI.git
cd CRH-AI

# 安装依赖
pip install -e .
```

---

## 🎯 快速开始

### 基础使用

```python
from crh_ai import CRHLLMBridge

def my_llm(prompt):
    return "基于几何锚定的响应"

bridge = CRHLLMBridge(llm=my_llm)

# 基础推理
result = bridge.think("明天北京天气如何？")
print(f"自洽性分数: {result['coherence_score']:.4f}")
```

### 增强推理（推荐）

```python
from crh_ai import CRHLLMBridge

# 使用几何提示词增强 + 多尺度剪枝
result = bridge.think_enhanced("Explain geometric redshift")
print(f"复杂度: {result['complexity']:.2f}")
print(f"使用尺度: {result['scale']}")
print(f"增强提示:\n{result['enhanced_prompt']}")
```

### 动态学习

```python
from crh_ai import CRHLLMBridge

bridge = CRHLLMBridge()

# 添加上下文
bridge.add_context("Physics is the study of matter and energy")
bridge.add_context("Redshift is spectral line displacement")

# 从验证事实中学习
bridge.learn_from_feedback({
    'summary': '光速约为 3×10^8 m/s',
    'type': 'physical',
    'region': 'physics',
    'confidence': 0.99
})
```

---

## 📁 项目结构

```
CRH-AI/
├── src/
│   └── crh_ai/
│       ├── core/                    # 核心几何引擎
│       │   ├── manifold.py          # CRH信息流形 + 动态ρ更新
│       │   ├── geometry.py          # 曲率、距离、测地线
│       │   ├── entanglement.py      # 纠缠梯度场
│       │   └── entropy_balance.py   # 熵平衡与自洽性
│       ├── constraints/             # 物理先天约束
│       │   ├── causal.py
│       │   ├── topological.py
│       │   └── physical_prior.py
│       ├── reconstruction/          # 弱相干重构
│       │   └── mapper.py
│       ├── integration/             # 大模型对接层
│       │   └── llm_bridge.py        # CRHLLMBridge
│       ├── prompts/                 # 几何提示词引擎（新增）
│       │   └── geometric_prompt_engine.py
│       ├── utils/                   # 工具函数
│       │   └── visualization.py
│       ├── core_base.py             # 核心集成类
│       └── __init__.py
├── tests/
│   ├── test_crh_geometric_base.py
│   └── test_enhancements.py         # 三大增强功能测试
└── pyproject.toml
```

---

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 单独运行增强功能测试
python tests/test_enhancements.py
```

---

## 📚 核心 API

### CRHLLMBridge

| 方法 | 说明 |
|------|------|
| `think(user_input)` | 标准推理流程 |
| `think_enhanced(user_input)` | 增强推理（几何提示 + 多尺度剪枝） |
| `learn_from_feedback(verified_fact)` | 从验证事实中学习（动态ρ更新） |
| `add_context(context_text, context_id)` | 添加几何上下文 |
| `clear_context()` | 清空上下文 |
| `get_system_state()` | 获取系统状态 |
| `get_geometric_report(geo_state)` | 生成几何状态报告 |

### CRHGeometricBase

| 方法 | 说明 |
|------|------|
| `embed(input_data, input_id)` | 嵌入输入到几何空间 |
| `perceive(query, context)` | 感知纠缠梯度场 |
| `check_constraints(geometric_reprs)` | 检查物理约束 |
| `process(input_data)` | 完整处理流程 |
| `update_from_feedback(verified_fact)` | 动态ρ更新 |
| `estimate_complexity(query)` | 估算问题复杂度 |
| `get_scaled_manifold(complexity)` | 获取多尺度流形视图 |

### GeometricPromptEngine

| 方法 | 说明 |
|------|------|
| `enhance_prompt(user_query, geo_state)` | 生成带几何描述子的提示 |
| `generate_geometry_report(geo_state)` | 生成详细的几何状态报告 |

---

## 🎯 实现优先级

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 时空几何锚定层 + 简单纠缠梯度 + LLM Bridge | ✅ 完成 |
| Phase 2 | 物理约束层 + 自洽性实时评估 | ✅ 完成 |
| Phase 3 | 弱相干重构 + 动态拓扑记忆 | ✅ 完成 |
| Phase 4 | 动态ρ更新 + 几何提示 + 多尺度剪枝 | ✅ 完成 |
| Phase 5 | 量子加速版本（可选） | ⏳ 规划中 |

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

