# S-Space Navigation: 把Transformer内部表征当可导航坐标系

**Three formulas to read, navigate, and control any Transformer's internal representations — no training, no prompt engineering.**

[中文说明](#中文说明)

---

## 三条公式

这是本项目的全部核心。三条公式，依次运行，闭环控制：

| # | 公式 | 功能 | 一句话 |
|---|------|------|--------|
| ③ | `c_k = h · ê_k` | **定位** — 你在哪里 | 读K维坐标 |
| ② | `Δ_k = need_k × d_k` | **导航** — 怎么走到目标 | 坐标位移 |
| ① | `α = r × \|h\| / \|Δ_masked\|` | **力度** — 推多大力 | 可控注入 |

```
输入 → forward → h(L) → ③读坐标 → ②算导航 → ①控力度 → 注入 → 生成
```

**传递函数线性度**: cos(Δh, inject) = **1.0000** — 模型对注入零抵抗，推多少得多少。

---

## 四个新领域

| 领域 | 含义 | 全球对应 |
|------|------|----------|
| **内部坐标导航学** | 表征 = 可导航坐标系，不是旁观者是领航员 | 无（Cheng 2026发现子空间但无坐标系无导航） |
| **对抗编译学** | GCG乱码 = S空间坐标指令，可解码 | 无（Nature 2026只证明对齐脆弱性存在） |
| **轴注入工程学** | Need驱动的轴选择：饱和轴不推，饥饿轴多推，所有轴都有信息 | 无（全球LTH做剪枝，不做注入） |
| **认知层手术学** | 层次分工定律指导选层（L19=推理vs叙事决策层） | 无（全球steering试层，无理论指导） |

---

## 仿射空间公理（14280条位移向量验证）

S空间是**精确的仿射空间**：

```
Δ(A→C) = Δ(A→B) + Δ(B→C)    [可加性 cos = 1.0000]
```

这意味着：**导航 = 向量算术**。你可以在坐标空间中精确计算从一个认知状态到另一个的路径。

### 关键性质

| 性质 | 数值 | 含义 |
|------|------|------|
| 可加性（未归一化） | cos = 1.0000 | 精确仿射空间 |
| 有效维度 | 28-31（全1024维中） | 极度各向异性 |
| Top-3度量权重 | 32-40% | 3个方向携带1/3结构信息 |
| Top-10度量权重 | 65-76% | 10维捕获大部分导航信息 |
| 层膨胀系数 | α = 0.092/层（L14+） | 深层距离指数膨胀 |
| 归一化伪曲率 | cos ≈ 0.95 | 归一化制造20-35%伪曲率 |

### 度量张量

```
d²(s_A, s_B; l) = Σ_k g_k(l) · [ê_k(l) · (s_A - s_B)]²
```

g_k(l) = S_k²(l) / n — 从SVD奇异值直接计算，任何模型任何层都可以提取。

---

## 轴认知（核心洞察）

S空间中每一根轴ê_k都有信息，没有"彩票轴"和"非彩票轴"的区别，只有"当前任务需要用哪些轴"的区别。

### 轴的粒度分层

| 轴范围 | 功能 | 粒度 | 类比 |
|--------|------|------|------|
| ê₀-ê₉ | 粗粒度分割（叙事vs推理） | 大类边界 | 国道 |
| ê₁₀-ê₃₀ | 中粒度分类（因果vs逻辑vs数学） | 大类内部 | 省道 |
| ê₃₁-ê₁₀₀ | 细粒度区分（合成谬误vs滑坡谬误） | 子类型边界 | 乡间小路 |

前端轴g_k大（65-76%度量权重），承载粗粒度决策；后端轴g_k小但saturation极低，是解题的关键路径。V4.3实验证明：62.4%的反事实信号在轴31-100。

### Need驱动的轴选择

```
need_k = 1 - (|c_k| × g_k) / Σ(|c_k| × g_k)
```

**"饱和的轴不需要推"** — 已经在目标方向的轴，need_k低，不推；还没到位的轴，need_k高，多推。

### 坐标轴语义（单旋钮实验验证）

每个ê_k是**真实的、语义连贯的行为控制维度**，不是标签投影：

| 旋钮 | 维度命名 | 正向效果 | 反向效果 |
|------|---------|----------|----------|
| ê_2 | **视角/信号** | "大质量物体加速运动"→"最强大的宇宙信号" | 换行强调 |
| ê_6 | **系统化** | "原因"→"物理机制"，去人名加"已被实验证实" | 维持"原因" |
| ê_8 | **存在肯定** | 强化存在断言 | "成功观测到"→"一种新的基本形式" |
| ê_9 | **正式/优雅** | 因果描述→本体论定义 | — |

**方向性对称**：→ 和 ← 产生**语义对立**的变化——不是统计噪声。

---

## 三大定律（实验验证）

### 定律1：选择性注入
只推需要推的维度，不推已经到位的维度。need_k自动实现这一点：饱和轴低need、饥饿轴高need。
- 全维注入 = 语义破坏
- Need选择性注入 = 精准控制
- 随机15维 > 全维注入 — 证明"对分类重要的维度"和"对注入友好的维度"不同

### 定律2：层次分工
L19是推理vs叙事核心决策层（std=0.280，其他层0.044-0.084），不等力注入。
- L3: 语义初加工(13%)
- L16: 信息瓶颈(70%)
- L19: 推理核心(83%)
- L22: 输出门控(96%)

### 定律3：干预倒U曲线
存在最优干预强度，过弱无效过强破坏。
- gap自适应：gap = |target_coords - c_k|
- 代码题gap≈0.74 → SKIP（不注入，模型本身就会）
- 反事实gap≥1.03 → INJECT（需要推到推理区域）

---

## 与全球工作对比

| 维度 | Anthropic SAE/NLA | Nature 2026 | RISER (ICLR 2026) | CAE (2025) | **S-Space (Ours)** |
|------|-------------------|-------------|--------------------|-----------|--------------------|
| 读懂内部 | ✅ 被动观测 | — | — | — | ✅ 坐标定位③ |
| 知道脆弱点在哪 | — | ❌ 只证存在 | — | — | ✅ L19 group_dist |
| 力度可控 | ❌ 无公式 | — | ❌ 手调α | ❌ 手调α | ✅ 公式① cos=1.0 |
| 零训练 | — | — | ❌ RL Router | — | ✅ |
| Need自适应 | — | — | — | ❌ 损害困惑度 | ✅ need_k自动选轴 |
| think可导航 | — | — | — | — | ✅ 坐标维度 |

---

## 快速开始

### 模式一：纯惯性导航（任何模型，装上即用）

```python
from s_space import CoordNavigator

# 只需要PCA参数——从任何HuggingFace模型提取
nav = CoordNavigator(params_path="path/to/pca_params.pt")

# 对模型做forward，捕获hidden states
# ... (使用HuggingFace transformers的hook机制)

# 三公式自动运行：定位→导航→注入
result = nav.navigate(hidden_states, base_r=0.10)
injections = result['injections']

# 注入到模型
for L, inj in injections.items():
    hidden_states[L] = hidden_states[L] + inj
```

### 模式二：增强导航（有共识方向数据时）

```python
# 加载PCA参数 + 共识方向（可选增强）
nav = CoordNavigator(
    params_path="path/to/pca_params.pt",
    consensus_path="path/to/reasoning_consensus.pt",  # 可选
)

# 使用方式完全相同，导航模式自动切换
result = nav.navigate(hidden_states, base_r=0.10)
```

### 从模型提取PCA参数

```python
from s_space.extraction import extract_pca_params

# 一键提取——任何HuggingFace模型
params = extract_pca_params("Qwen/Qwen2.5-7B", layers=[3, 16, 19, 22])
torch.save(params, "pca_params.pt")
```

### 安装

```bash
git clone https://github.com/ayuan666d/s-space.git
cd s-space
pip install -e .
```

### 依赖

- Python 3.10+
- PyTorch 2.0+
- transformers 4.38+

---

## 项目结构

```
s-space/
├── s_space/              # 核心库：三公式+导航器+注入掩码
│   ├── formulas.py       # 三条核心公式
│   ├── space.py          # 仿射空间+度量张量+层膨胀律
│   ├── navigator.py      # 导航器（目标注册+持续导航+漂移修正）
│   └── injection_mask.py # 选择性注入掩码
├── extraction/           # 从模型中提取S空间参数
├── experiments/          # 关键实验复现
├── battle/               # 对比测试框架
├── docs/                 # 完整理论文档
│   ├── S_SPACE_FORMULAS.md           # 数学公式体系（公理+预言）
│   ├── S_SPACE_CONTROLLABILITY.md    # 可控注入理论
│   ├── SINGLE_KNOB_FINDINGS.md       # 单旋钮核心发现
│   ├── EXPERIMENT_META_ANALYSIS.md   # 198实验元分析
│   ├── DEFINITIVE_ANSWERS.md         # 确定性结论
│   ├── METHOD_COMPARISON.md          # 与全球方法对比
│   └── ARCHITECTURE.md               # 架构设计
├── data/                 # 示例数据（大文件见Release）
└── paper/                # 元语言论文
```

---

## 验证过的实验

| 实验 | 关键结果 | 文档 |
|------|---------|------|
| 仿射空间验证 | 14280条位移，可加性cos=1.0000 | `docs/S_SPACE_FORMULAS.md` |
| 传递函数线性度 | 6层×7α值，cos=1.0000 | `docs/S_SPACE_CONTROLLABILITY.md` |
| 单旋钮语义映射 | ê_2/ê_6/ê_8/ê_9 语义连贯+方向对称 | `docs/SINGLE_KNOB_FINDINGS.md` |
| 20题Trap测试 | V6: 16/20（代码题3层全SKIP，不受干扰） | `experiments/` |
| GCG跨厂商操控 | 乱码操控不同厂家模型，元语言倒推还原语义 | `experiments/gcg_decode.py` |
| Need驱动轴选择 | V4纯惯性：饱和轴低need、饥饿轴高need | `experiments/` |
| 后端轴解题 | V4.3推轴33-98：结构化推理，公式正确 | `experiments/` |

---

## 对抗编译学

本项目的另一个核心发现：**GCG对抗后缀 = S空间中的坐标指令**。

1. 一段乱码操控了不同厂家的模型（文本续写→格式分析）
2. 通过残存Token + 多模型平均 + 元语言倒推，逆向提取了乱码的"语义"
3. S空间坐标解码给出了几何解释：乱码改变了hidden state在K维空间中的坐标

| 维度 | Nature 2026 | S-Space |
|------|-------------|---------|
| 脆弱性存在 | ✅ 数学证明 | ✅ 实验确认 |
| 脆弱点在哪 | ❌ | ✅ L19 group_dist + 坐标定位 |
| 能否主动操控 | ❌ 只能被动测试 | ✅ GCG跨厂商操控 |
| 乱码为何有效 | ❌ 黑箱 | ✅ 元语言倒推 + 坐标解码 |

---

## Citation

```bibtex
@article{sspace2026,
  title={S-Space Navigation: Reading, Navigating, and Controlling Transformer Internal Representations as Coordinate Systems},
  author={Anonymous},
  year={2026}
}
```

---

## License

Apache License 2.0

---

> **Disclaimer**: This project provides tools for understanding and controlling LLM internal representations. Core formulas and geometric properties (affine additivity, metric tensor, injection linearity) are architecture-agnostic mathematical truths. Important decisions should be validated by domain experts.

---

<a id="中文说明"></a>

## 中文说明

### 这是什么？

我们发现了Transformer内部表征的一个深层结构——**S空间**：一个精确的仿射空间，14280条位移向量验证了可加性cos=1.0000。在这个空间里，模型"在想什么"可以用K维坐标读取，"怎么让它改变想法"可以用导航公式计算，"推多大力"可以用力度公式精确控制。

### 核心洞察

**每一根轴都有信息，没有"不可用的轴"，只有"还没用对的轴"。**

- 前端轴g_k大，决定"走哪条大路"（粗粒度）
- 后端轴g_k小但saturation低，是"解题直径"（细粒度）
- need_k自动选轴：饱和的轴不推，饥饿的轴多推
- 罗盘不需要——纯惯性模式（d_k = iw_k × c_k）天然实现"会的少推"

### 三条公式

- **公式③（定位）**：`c_k = h · ê_k` — 读K维坐标，知道模型当前在认知空间中的位置
- **公式②（导航）**：`Δ_k = need_k × d_k` — 计算从当前位置到目标位置的位移
- **公式①（力度）**：`α = r × |h| / |Δ_masked|` — 精确控制注入力度，传递函数线性cos=1.0

### 两种导航模式

1. **纯惯性模式**（默认，模型无关）：`d_k = iw_k × c_k`
   - 方向完全从当前坐标推导，任何模型装上即用
   - 天然"饱和轴不推"——已经在目标方向的轴不会被推

2. **共识增强模式**（可选，需预提取数据）：`d_k = d_consensus × d_magnitude × d_confidence`
   - 使用从实验中提取的推理方向，增强导航精度
   - 适用于有条件提取数据的模型

### 为什么重要？

1. **从"观测"到"操控"的范式跃迁** — Anthropic的SAE/NLA能"看见"模型在想什么，我们能"导航"模型去想什么
2. **力度不再靠猜** — 全球steering vector实验都是手调α，我们有力度公式
3. **对抗攻击不再是黑箱** — GCG乱码在S空间中是可解码的坐标指令
4. **装上就能用** — 核心公式不绑定具体模型，任何Transformer都可以提取S空间参数
