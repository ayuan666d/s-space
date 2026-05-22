# 所有方法对比分析

## 📊 方法演进时间线

```
2026-05-10 ~ 2026-05-11: 路由校准系列 (v1-v8)
2026-05-12:             自进化路由器 (v1-v7) + run_final.py
2026-05-14:             SmartRouter v8/v10
2026-05-15:             动态注入引擎 (dynamic_inject.py)
2026-05-16:             S空间导航器 (s_space_loop_navigator.py)
```

---

## 🔬 核心方法分类

### **类别1：路由校准系列（Route Calibration v1-v8）**

**目标**：训练一个小型网络，学习如何路由到正确的认知模式

| 版本 | 核心思路 | 训练方式 | 效果 | 问题 |
|------|---------|---------|------|------|
| v1-v5 | 纯hidden state对齐 | 只优化hidden state距离 | 6/20 → 13/20 | 模型崩坏，输出质量下降 |
| v6 | hidden state only | 只在彩票维度上对齐 | 1/20 (5%) ❌ | 完全失败！LM loss被忽略 |
| **v7** | **联合训练** | **0.7×LM_loss + 0.3×HS_loss** | **待验证** | **需要微调权重** |
| v8 | - | - | - | - |

**关键发现：**
> v6证明了：只优化hidden state会导致模型崩溃！必须同时考虑语言建模损失。

**代码特征（run_route_calibration_v7.py）：**
```python
# 联合损失
total_loss = beta_lm * lm_loss + beta_hs * (hs_loss / 2)
# beta_lm=0.7, beta_hs=0.3

# 只训练彩票维度
for k, tp in target_params.items():
    tp['param'].grad.data[~tp['mask']] = 0  # 梯度掩码
```

---

### **类别2：每层独立注入训练（run_final.py）**

**目标**：为每层学习独立的注入向量v[l]

**核心思想：**
- 不是用固定的方向d_l，而是学习每层的v[l]
- 每层有自己的SGD优化器
- COS目标：最大化cos(v[l], d_l)

**训练流程：**
```python
# 1. 每层初始化v[l] = d_l * 0.3
v = {l: torch.zeros(H, requires_grad=True) for l in ACT}

# 2. 每层独立优化器
opt = {l: torch.optim.SGD([v[l]], lr=1e-3) for l in ACT}

# 3. 200轮训练，每层建自己的小图
for ep in range(200):
    for l in ACT:
        loss_l = 0
        for idx in range(len(qa)):
            raw = qa[idx][l].float().to(d)
            ld = ldim[l]  # 15个彩票维度
            dl = src[l].float()[ld]
            
            # 只推15维
            corr = raw[ld].float() + v[l].float()[ld] * 3.0
            cos = (corr * dl).sum() / (corr.norm() * dl.norm() + 1e-8)
            
            loss_l += -cos  # 最大化cos
        
        opt[l].zero_grad()
        loss_l.backward()
        opt[l].step()
```

**测试结果：**
- 之前运行过，但具体得分未记录
- 理论上应该比固定方向好，因为针对任务优化了

**优势：**
- ✅ 每层独立优化，互不干扰
- ✅ 稀疏注入（只推15维）
- ✅ COS自省（最大化与方向的相似度）

**劣势：**
- ❌ 需要预捕获所有hidden state（内存占用大）
- ❌ 没有类型区分（所有题用同样的v[l]）

---

### **类别3：动态注入引擎（dynamic_inject.py）**

**目标**：不同生成阶段使用不同的注入策略

**三层架构：**
```
1. CoordinateMap — 每层的目标坐标
2. InjectionSchedule — 动态调度策略
3. DynamicInjector — 执行引擎
```

**层角色定义：**
```python
ROLES = {
    0:  'mode_entry',      # 模式入口
    3:  'recognition',     # 识别增强
    7:  'narrative',       # 叙事瓶颈
    13: 'deep_reason',     # 深层推理开始
    19: 'code_structure',  # 代码/结构
    23: 'output_gate',     # 输出门控
}
```

**预设方案（Profiles）：**
```python
PROFILES = {
    'recognition': {
        INIT:  {0: 1.5, 1: 1.2, 3: 2.0},   # 前3个token强L3
        EARLY: {3: 1.8, 4: 1.0, 7: 0.3},
        MID:   {3: 1.2, 11: 0.5, 13: 0.3},
        LATE:  {19: 0.5, 23: 0.3},
    },
    'code': {...},
    'logic': {...},
    'detail': {...},
}
```

**优势：**
- ✅ 细粒度控制（每个token阶段不同）
- ✅ 基于实验数据（逐层注入结果）
- ✅ 灵活（可自定义profile）

**劣势：**
- ❌ 复杂度高（需要维护多个profile）
- ❌ 需要手动调参（每个阶段的力度）
- ❌ 没有自适应能力（固定schedule）

---

### **类别4：自进化路由器（self_evolving_router v1-v7）**

**目标**：类型感知 + 记忆系统 + COS自省循环

**v7核心架构：**
```
问题 → TypeClassifier → StrategyRouter → COS自省循环 → 输出 → 记忆更新
```

**类型策略表：**
```python
STRATEGY = {
    '数学':   {'mode': 'zero',  'strength': 0.0},   # 不注入
    '逻辑':   {'mode': 'zero',  'strength': 0.0},   # 不注入
    '翻译':   {'mode': 'zero',  'strength': 0.0},   # 不注入
    '叙事':   {'mode': 'minus', 'strength': -0.25}, # -d释放创造力
    '常识':   {'mode': 'auto',  'strength': 0.15},  # COS自省
    '因果':   {'mode': 'auto',  'strength': 0.15},
    '知识':   {'mode': 'plus',  'strength': 0.25},  # +d结构化
    '道德':   {'mode': 'plus',  'strength': 0.20},
    '反事实': {'mode': 'plus',  'strength': 0.20},
}
```

**COS自省循环：**
```python
for attempt in range(3):
    t = generate(question, strengths, layers, negative)
    v = valid(t)
    ds = ds_judge(question, t) if v else 0
    
    if ds == 1:  # 成功
        break
    
    # 失败则自适应调整
    if ds == 0:
        strengths = {l: s * 1.2 for l, s in strengths.items()}
```

**记忆系统：**
```python
# 保存成功的经验
memory.append({
    'type': qtype, 
    'e': embedding, 
    's': strengths,
    'cos': 0, 
    'q': quality_score, 
    'el': layers
})

# 下次匹配
if cos(current_emb, mem_emb) > 0.8:
    reuse(mem_strengths)  # 直接复用
```

**测试结果（刚刚运行）：**
- **Score: 15/20 (75%)** ⭐
- zero模式：11/14 (79%)
- auto模式：4/6 (67%)

**优势：**
- ✅ 类型感知（不同题型不同策略）
- ✅ COS自省（动态调整强度）
- ✅ 记忆系统（经验积累）
- ✅ 实际效果最好（75%）

**劣势：**
- ❌ 依赖DeepSeek API做质量判断（有成本）
- ❌ 分类器需要训练（虽然只需一次）
- ❌ minus/plus模式未充分验证

---

### **类别5：S空间导航器（s_space_loop_navigator.py）**

**目标**：基于S空间几何理论的循环检测和导航

**核心组件：**
```python
1. SSpaceNavigator - 定位+导航
   - get_current_position() - 计算到各质心的cosine
   - is_in_danger_zone() - 检测是否在危险区域
   - get_safe_displacement() - 获取导航位移

2. SSpaceLoopDetector - 循环检测
   - detect_from_trajectory() - 从轨迹检测循环
   - _check_position_stability() - 位置稳定性检测

3. SSpaceLoopPreventionEngine - 预防引擎
   - generate_with_s_navigation() - S空间导航生成
```

**危险区域定义：**
```python
dangerous_categories = {'百度知道', '简单问答', '选择题模板', '重复模式'}
safe_categories = {'因果推理', '逻辑分析', '深度思考', '结构化回答'}
```

**优势：**
- ✅ 理论基础最强（S空间几何）
- ✅ 实时监控（持续捕获隐藏状态）
- ✅ 预防性（在陷入循环前干预）

**劣势：**
- ❌ 实现未完成（_gen_with_injection是空的）
- ❌ 需要大量S空间数据（s_raw_full.pt, s_displacements_full.pt）
- ❌ 计算开销大（每步都要计算cosine到所有质心）

---

## 📈 效果对比汇总

| 方法 | 得分 | 循环率 | 是否需要训练 | 复杂度 | 成熟度 |
|------|------|--------|------------|--------|--------|
| Baseline | ~11/20 | 45% | ❌ | 低 | ✅ |
| Route Cal v6 | 1/20 | 95% ❌ | ✅ (失败) | 中 | ❌ |
| Route Cal v7 | 待验证 | - | ✅ (联合) | 中 | 🔄 |
| run_final.py | 未知 | - | ✅ (分层) | 中 | 🔄 |
| dynamic_inject | 未测试 | - | ❌ | 高 | 🔄 |
| **SmartRouter v7** | **15/20** | **25%** | ✅ (分类器) | 中 | ✅ |
| SNav (未完成) | - | - | ❌ | 高 | ❌ |

---

## 💡 关键洞察

### **1. 什么方法有效？**

✅ **有效的要素：**
- 类型感知（不同题型不同策略）
- COS自省循环（动态调整强度）
- 记忆系统（经验积累）
- 稀疏注入（只推15维）

❌ **无效的要素：**
- 均匀注入（所有层同样力度）
- 预定义固定目标（强制推到某个值）
- 只优化hidden state（忽略LM loss）
- 全维度注入（推1024维）

### **2. 为什么SmartRouter v7最有效？**

1. **类型策略表** - 数学/逻辑不需要注入，避免了破坏基线能力
2. **COS自省** - 不是推到固定值，而是迭代直到满足条件
3. **记忆系统** - 成功的经验被永久保存
4. **简单有效** - 不需要复杂的S空间计算

### **3. 各方法的适用场景**

| 方法 | 最佳场景 |
|------|---------|
| SmartRouter v7 | **通用推理任务**（当前最优） |
| dynamic_inject | 需要细粒度控制的场景（如代码生成） |
| run_final.py | 特定任务的离线优化 |
| SNav | 长文本生成（需要防循环） |
| Route Cal v7 | 需要微调权重的场景（但风险高） |

---

## 🚀 下一步建议

### **优先级1：完善SmartRouter v7**
- 测试minus/plus模式（找叙事/道德类题目）
- 优化auto模式的COS目标值
- 减少对外部API的依赖（本地质量判断）

### **优先级2：整合dynamic_inject的profile思想**
- 把dynamic_inject的层角色定义整合到SmartRouter
- 让不同题型使用不同的注入schedule

### **优先级3：完成SNav的实现**
- 补全_gen_with_injection方法
- 在长文本任务上测试防循环效果

### **优先级4：放弃Route Calibration系列**
- v6已经证明失败
- v7需要微调，风险高
- 不如专注零训练的方法

---

## 🎯 最终推荐方案

**SmartRouter v7 + dynamic_inject的profile思想**

```
问题 → 类型分类 → 选择profile(recognition/code/logic/detail)
     → COS自省循环(3次迭代)
     → 记忆匹配(命中则复用)
     → 输出
```

**理由：**
1. 已验证有效（15/20 = 75%）
2. 零训练成本（只需要分类器，一次训练即可）
3. 可扩展（容易添加新类型和新profile）
4. 理论清晰（类型感知 + COS自省）
