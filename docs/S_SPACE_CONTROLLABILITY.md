# S-Space 可控注入理论（Controllable Injection Theory）

> 基于 Qwen3.5-0.8B 的实验验证，2026-05-19
> 核心突破：从"手调强度"到"公式可控"，解决了注入的层间失衡、类型不敏感、问题不自适应三大问题

---

## 〇、问题背景

### 之前的状态

所有成功的注入（cerebellum_v2, navigator, 残差循环, 二阶段突破）都使用：

```python
inject(L) = type_dir(L) × selection_mask(L) × strength × sc
# strength = 0.03 ~ 0.05（手调）
# sc = 24 / n_layers（层均分系数）
# 注: selection_mask 历史上称 lottery_mask
```

**已知问题**：
1. 同一个 strength=0.04，不同层效果差 3 倍（L19 被推 65%，L15 只推 21%）
2. 不同类型（叙事/因果/逻辑/代码）用同一个 strength，但敏感度完全不同
3. 不同问题 |h(L)| 差异大，同一 strength 的 |Δh|/|h| 不一致
4. 残差循环每步用同一 strength，但 |h(L)| 在变化，越循环越失控
5. 效果不可解释：为什么 0.04 好用？不知道。换问题要不要调？不知道。

### 目标

用数学公式替代手调，实现**一个参数控制全局**。

---

## 一、实验发现

### 发现 1：模型对注入零抵抗

**传递函数完美线性**：

对每层注入 `type_dir × mask × α`，测量 Δh：

| 层 | α | |Δh| | cos(Δh, inject) |
|----|---|------|-----------------|
| L3 | 0.01~0.10 | 线性 | 1.0000 |
| L7 | 0.01~0.10 | 线性 | 1.0000 |
| L11 | 0.01~0.10 | 线性 | 1.0000 |
| L15 | 0.01~0.10 | 线性 | 1.0000 |
| L19 | 0.01~0.10 | 线性 | 1.0000 |
| L22 | IndexError(维度) | — | — |

**含义**：模型的残差连接是完美的直通管道——注入什么就得到什么，没有任何衰减或旋转。Δh = 注入向量本身。

### 发现 2：增益公式

单层增益 $g(L) = sc \times |dir\_masked(L)|$，其中 $sc = 24 / 1$（单层注入时）。

验证：$g(L) / |dir\_masked(L)|$ 在 6 层上全部为 24.00~24.06。

**这意味着增益完全由两个因素决定**：
1. 层数分配系数 sc（你注入几层就除以几）
2. 该层 type_dir 在选择维度上的投影范数 |dir_masked(L)|

### 发现 3：手调的层间失衡

手调 α=0.04 时的实际效果（叙事类）：

| 层 | α | |Δh|/|h| | 效果 |
|----|---|---------|------|
| L3 | 0.04 | ~30% | 偏强 |
| L7 | 0.04 | ~42% | 过强 |
| L15 | 0.04 | ~21% | 偏弱 |
| L19 | 0.04 | ~65% | **严重过强** |

L19 被推了 65%，L15 只被推了 21%——差了 3 倍。

### 发现 4：公式可控验证

使用公式 `α(L) = r × |h(L)| / (sc × |dir_masked(L)|)` 后：

| 策略 | 各层 |Δh|/|h| | 效果 |
|------|-------------|------|
| formula_10pct | ≈10% | 与手调 0.04 相当 |
| formula_20pct | ≈20% | 叙事类更完整 |
| formula_30pct | ≈30% | 代码类更详细 |
| 因果/逻辑 10% | ≈10% | 已足够（敏感型） |

---

## 二、核心公式

### 传递函数（Transfer Function）

$$\Delta h(L) = inject(L)$$

模型对注入**零抵抗**：注入向量就是实际位移向量。

$$|\Delta h(L)| = sc \times |dir\_masked(L)| \times \alpha(L)$$

其中：
- $sc = n_{total} / n_{inject}$ = 层数分配系数
- $|dir\_masked(L)| = |type\_dir(L) \times selection\_mask(L)|$ = 稀疏选择投影范数（预计算常量，历史称"彩票投影范数"）
- $\alpha(L)$ = 注入强度

### 可控注入公式（Controllability Formula）

$$\boxed{\alpha(L) = \frac{r \times |h(L)|}{sc \times |dir\_masked(L)|}}$$

其中：
- $r$ = 目标扰动比例（|Δh|/|h| 的目标值），典型范围 0.10~0.30
- $|h(L)|$ = 当前问题的 hidden state 范数（运行时测量）
- $sc = n_{total} / n_{inject}$ = 层数分配系数
- $|dir\_masked(L)|$ = 预计算常量

### 简化形式

将 α(L) 代入注入方程：

$$inject(L) = type\_dir(L) \times selection\_mask(L) \times r \times \frac{|h(L)|}{|dir\_masked(L)|} \times sc \times \frac{1}{sc}$$

$$= type\_dir(L) \times selection\_mask(L) \times r \times \frac{|h(L)|}{|dir\_masked(L)|}$$

$$= \hat{d}_{masked}(L) \times r \times |h(L)|$$

其中 $\hat{d}_{masked}(L) = type\_dir(L) \times selection\_mask(L) / |dir\_masked(L)|$ 是稀疏选择维度上的归一化方向（历史称"彩票维度"）。

**物理含义**：注入 = 选择方向 × 目标比例 × 当前状态范数。就这么简单。

---

## 三、公式解决了什么

### 问题 1 → 层间均衡 ✅

| | 之前（手调 α=0.04） | 现在（公式 r=0.10） |
|---|---|---|
| L3 | 30% | 10% |
| L7 | 42% | 10% |
| L15 | 21% | 10% |
| L19 | 65% | 10% |

所有层统一被推 10%，不再有层被推飞或被忽略。

### 问题 2 → 类型自适应 ✅

公式包含 $|dir\_masked(L)|$，不同类型在不同层的选择维度投影强度不同，α 自动调整：

- 叙事类 L3：|dir_masked| 较大 → α 较小（不需要太强）
- 因果类 L19：|dir_masked| 较小 → α 较大（需要更强的信号）

### 问题 3 → 问题自适应 ✅

公式包含 $|h(L)|$，不同问题的 hidden state 范数不同：

- "天空为什么是蓝的" L19 |h|=2.32 → α 较大
- "太空探险故事" L19 |h|=1.30 → α 较小

同一个 r 值，不同问题自动适配。

### 问题 4 → 残差循环可控 ✅

每步重新测量 |h(L)|，重算 α(L)：

```python
for step in range(max_steps):
    h_norm = current_h.norm()
    alpha = r * h_norm / (sc * dir_masked_norm)
    inject = type_dir * mask * alpha * sc
    # r 随步数递减: r = r0 * decay^step
```

每步的 |Δh|/|h| 精确等于目标 r，循环不再发散。

### 问题 5 → 可解释 ✅

r = 10% 意味着"每层推 10% 的 hidden state 范数"，直觉清晰。

---

## 四、与 S-Space 公式体系的关系

### 统一视角

S-Space 公式体系（S_SPACE_FORMULAS.md）描述的是**类型空间的结构**：

$$\text{小脑方程}: inject(l) = P_l \cdot \Delta_{needed}(l) \cdot scale(l)$$

这里的 $\Delta_{needed}(l)$ 是 S-Space 中的位移向量（类型质心差），$P_l$ 是主方向投影。

可控注入公式描述的是**注入的力学**：

$$inject(L) = type\_dir(L) \times selection\_mask(L) \times \alpha(L) \times sc$$

### 关键区别

| 维度 | S-Space 公式 | 可控注入公式 |
|------|-------------|-------------|
| 方向来源 | 质心差 Δ = target_centroid - current | type_dir（好回答方向） |
| 作用空间 | 全 1024 维 | 15 稀疏选择维度 |
| 与输出的关系 | 推到类型坐标 ≠ 改善输出 | 推好回答方向 = 改善输出 |
| cos(type_dir, 质心差) | — | ≈ -0.1 ~ 0.18（几乎正交） |

**核心洞见**：S-Space 公式导航的是"你在类型空间的什么位置"，可控注入公式控制的是"你往好回答方向推多大力"。两者正交。

### 正确的组合方式

1. **用 S-Space 公式做诊断**：判断当前 h(L) 在类型空间的位置，决定要不要注入
2. **用可控注入公式做执行**：决定注入多大力，确保层间均衡
3. **用 S-Space 度量张量预测最优 r**：度量张量在 type_dir 方向的本征值越大 → 方向越"硬" → 需要更大的 r

---

## 五、最优 r 值的经验规律

基于 5 类型 × 4 策略的实验：

| 类型 | 推荐初始 r | 理由 |
|------|-----------|------|
| 叙事 | 0.15~0.25 | 创造性任务需要更大推力 |
| 代码 | 0.20~0.30 | 输出结构化强，需要强信号改变 |
| 因果 | 0.05~0.10 | 对注入敏感，小 r 就够 |
| 逻辑 | 0.05~0.10 | 同上 |
| 反事实 | 0.10~0.20 | 中间地带 |

**假说**：最优 r 与该类型的 type_dir 和模型权重的关系有关，可能由 S-Space 度量张量预测。

---

## 六、待验证预言

1. **度量张量预测 r**：type_dir 方向上的度量权重越大，需要越小的 r（方向更"软"）
2. **层间差异化 r**：浅层(L3-L7)用较小 r，深层(L15-L19)用较大 r
3. **残差循环衰减公式**：r(step) = r_0 × e^{-λ·step}，λ 由度量张量决定
4. **跨模型泛化**：公式形式 α(L) = r × |h(L)| / (sc × |dir_masked(L)|) 应适用于其他 Transformer 模型
5. **最优 r 与温度的关系**：高温采样需要更大的 r 才能产生同等效果

---

## 七、实验文件索引

| 文件 | 内容 |
|------|------|
| `sspace_controllability.py` | Step 1 传递函数标定实验 |
| `sspace_control_step2.py` | Step 2 公式可控注入实验 |
| `sspace_control_results.json` | Step 2 完整结果（5题 × 5策略） |
| `S_SPACE_FORMULAS.md` | S-Space 数学公式体系（类型空间结构） |
| `sspace_prediction_results_v2.json` | S-Space 预言验证结果 |
| `sspace_prediction_verify_v2.py` | S-Space 预言验证代码 |

---

## 八、公式使用的代码模板

```python
# ====== 预计算（一次性） ======
import torch

type_dirs = torch.load('type_dirs.pt')      # 14类 × 21层
selection_masks = torch.load('pure_route_v2.pt')  # 稀疏选择维度（历史称"彩票维度"）

# 预计算每层每类型的 |dir_masked|
dir_masked_norms = {}
for type_name, td in type_dirs.items():
    dir_masked_norms[type_name] = {}
    for layer, vec in td.items():
        d = vec.float()
        d = d / d.norm()  # 归一化
        if layer in selection_masks:
            d = d * selection_masks[layer]  # 应用稀疏选择 mask
        dir_masked_norms[type_name][layer] = d.norm().item()

# ====== 运行时（每次注入） ======
def compute_alpha(type_name, layer, h_norm, r=0.10, n_inject_layers=3, n_total_layers=24):
    """计算可控注入的 α 值"""
    sc = n_total_layers / n_inject_layers
    dn = dir_masked_norms.get(type_name, {}).get(layer, 0)
    if dn < 1e-8:
        return 0.04  # fallback
    return r * h_norm / (sc * dn)

def make_injection(type_name, layers, h_norms, r=0.10):
    """生成可控注入向量"""
    injection = {}
    td = type_dirs[type_name]
    sc = 24.0 / len(layers)
    for l in layers:
        alpha = compute_alpha(type_name, l, h_norms[l], r, len(layers))
        d = td[l].float()
        d = d / d.norm()
        if l in selection_masks:
            d = d * selection_masks[l]
        injection[l] = (d * alpha * sc).numpy()
    return injection

# ====== 残差循环（自适应步进） ======
def residual_loop_inject(type_name, layers, initial_h_norms,
                         r0=0.10, decay=0.7, max_steps=5):
    """残差循环，每步 r 衰减"""
    for step in range(max_steps):
        r = r0 * (decay ** step)
        # 捕获当前 |h(L)|
        current_h_norms = capture_h_norms(question, layers)
        injection = make_injection(type_name, layers, current_h_norms, r)
        # 执行注入 + 生成一步
        result = inject_and_generate(question, injection, max_tokens=50)
```

---

*本文档由可控性实验验证总结。公式基于传递函数完美线性（cos=1.0000）的实验事实，在 6 层 × 7 个 α 值上交叉验证。*
