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
| **彩票注入工程学** | 彩票维度 = 精准注入通道（15维操控+1009维零副作用） | 无（全球LTH做剪枝，不做注入） |
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

## 坐标轴语义（单旋钮实验验证）

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

### 定律1：语义保护区
只推15维彩票维度，不动1009维。
- full_1024注入 = 2/20（语义被破坏）
- lottery_15注入 = 16/20（精准控制）
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
| 语义保护区 | — | — | — | ❌ 损害困惑度 | ✅ 15维+1009维 |
| think可导航 | — | — | — | — | ✅ 坐标维度 |

---

## 快速开始

```python
from s_space import CoordNavigator

# 加载导航器（需要预提取的PCA参数）
nav = CoordNavigator.from_pretrained("path/to/coord_nav_params.pt")

# 对模型做forward，捕获hidden states
# ... (使用HuggingFace transformers的hook机制)

# ① 读坐标
coords = nav.read_coords(h, layer=19)  # c_k = h · ê_k

# ② 算导航
delta_k = nav.compute_delta(coords, layer=19)  # Δ_k = need_k × d_k

# ③ 控力度
injection = nav.compute_injection(h, layer=19, delta_k, r_eff=0.10)
# α = r × |h| / |Δ_masked|，保证 |inject|/|h| = r

# 注入到模型
h_new = h + injection
```

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/s-space-navigation.git
cd s-space-navigation
pip install -e .
```

### 依赖

- Python 3.10+
- PyTorch 2.0+
- transformers 4.38+
- 1 GPU with 8GB+ VRAM

---

## 项目结构

```
s-space-navigation/
├── s_space/              # 核心库：三公式+导航器+彩票维度
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
| 水池题 | V4: 4.8h（Baseline ❌） | `experiments/` |

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

## 论文

元语言理论论文（Draft 4）见 `paper/` 目录。

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

> **Disclaimer**: This project provides tools for understanding and controlling LLM internal representations. All experiments were conducted on small models (0.8B). Scaling to larger models and standard benchmarks is ongoing work. Important decisions should be validated by domain experts.

---

<a id="中文说明"></a>

## 中文说明

### 这是什么？

我们发现了Transformer内部表征的一个深层结构——**S空间**：一个精确的仿射空间，14280条位移向量验证了可加性cos=1.0000。在这个空间里，模型"在想什么"可以用K维坐标读取，"怎么让它改变想法"可以用导航公式计算，"推多大力"可以用力度公式精确控制。

### 三条公式

- **公式③（定位）**：`c_k = h · ê_k` — 读K维坐标，知道模型当前在认知空间中的位置
- **公式②（导航）**：`Δ_k = need_k × d_k` — 计算从当前位置到目标位置的位移
- **公式①（力度）**：`α = r × |h| / |Δ_masked|` — 精确控制注入力度，传递函数线性cos=1.0

### 为什么重要？

1. **从"观测"到"操控"的范式跃迁** — Anthropic的SAE/NLA能"看见"模型在想什么，我们能"导航"模型去想什么
2. **力度不再靠猜** — 全球steering vector实验都是手调α，我们有力度公式
3. **对抗攻击不再是黑箱** — GCG乱码在S空间中是可解码的坐标指令
4. **通用性** — 公式不绑定具体模型，任何Transformer都可以提取S空间参数
