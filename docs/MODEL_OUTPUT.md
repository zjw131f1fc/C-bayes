# 模型输出规范 (model_output)

## 业务背景

**与星共舞 (Dancing with the Stars)** 是一档真人秀节目：
- 每季约12位**名人 (celebrity)** 参赛
- 每位名人搭配一位**专业舞者 (pro)**
- 每周表演后，观众投票 + 评委打分，排名最低者淘汰
- 比赛持续约10周，直到决出冠军

我们的模型预测：**每位名人每周能获得多少粉丝票占比 (P_fan)**

## 数据维度

### 观测 (observation)

一条**观测** = 一位名人在某一周的表演记录

```
观测示例：
- 观测0: 名人A 第1周 (搭档舞者X)
- 观测1: 名人B 第1周 (搭档舞者Y)
- ...
- 观测50: 名人A 第2周 (搭档舞者X)  ← 同一名人，不同周
```


## 数据结构

```python
datas['model_output'] = {
    # --- 维度信息 ---
    'n_obs': int,               # 观测数 (~400)
    'n_celebs': int,            # 名人数 (36)
    'n_pros': int,              # 舞者数 (15)

    # --- 观测索引 [n_obs] ---
    'celeb_idx': ndarray,       # 每条观测对应的名人ID
    'pro_idx': ndarray,         # 每条观测对应的舞者ID
    'week_idx': ndarray,        # 每条观测对应的周ID

    # --- 观测级预测 [n_obs] ---
    'mu': ndarray,              # 潜在投票强度
    'P_fan': ndarray,           # 粉丝票占比 (周内归一化)

    # --- 观测级分解 [n_obs] ---
    # 汇总
    'alpha_contrib': ndarray,   # 名人效应贡献 α[celeb_idx]
    'delta_contrib': ndarray,   # 舞者效应贡献 δ[pro_idx]
    'linear_contrib': ndarray,  # 线性特征贡献 β'X
    'spline_contrib': ndarray,  # 样条特征贡献 Σf(x)
    # 线性特征细分
    'linear_score_trend': ndarray,      # 分数趋势贡献
    'linear_teflon_factor': ndarray,    # teflon因子贡献
    ...
    # 样条特征细分
    'spline_z_score': ndarray,          # 评委分z-score贡献
    'spline_weeks_survived': ndarray,   # 存活周数贡献
    ...

    # --- 模型参数 ---
    'alpha': ndarray,           # [n_celebs] 名人效应
    'delta': ndarray,           # [n_pros] 舞者效应
    'beta_obs': ndarray,        # [n_linear] 线性系数
    'spline_coefs': list,       # 样条系数列表

    # --- 元信息 ---
    'feature_names': {
        'beta_obs': list,       # 线性特征名
        'splines': list,        # 样条特征名
    },
}
```

## 字段说明

| 字段 | 维度 | 含义 |
|------|------|------|
| `mu` | [n_obs] | 潜在投票强度，未归一化 |
| `P_fan` | [n_obs] | 粉丝票占比，同一周内所有选手的P_fan之和=1 |
| `alpha_contrib` | [n_obs] | 该观测中名人自身带来的贡献 |
| `delta_contrib` | [n_obs] | 该观测中舞者搭档带来的贡献 |
| `linear_contrib` | [n_obs] | 线性特征（分数趋势等）的贡献 |
| `spline_contrib` | [n_obs] | 样条特征（评委分、存活周数）的贡献 |

**关系：** `mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib`

## 使用示例

### 1. 查询某名人某周的预测

```python
out = datas['model_output']

# 名人5在第3周
mask = (out['celeb_idx'] == 5) & (out['week_idx'] == 3)
print(f"P_fan = {out['P_fan'][mask][0]:.3f}")
```

### 2. 绘制某名人的投票占比随时间变化

```python
celeb_id = 5
mask = (out['celeb_idx'] == celeb_id)
weeks = out['week_idx'][mask]
P_fan = out['P_fan'][mask]

plt.plot(weeks, P_fan, 'o-')
plt.xlabel('周')
plt.ylabel('粉丝票占比')
```

### 3. 分解某观测的贡献

```python
obs_idx = 10
contributions = {
    '名人效应': out['alpha_contrib'][obs_idx],
    '舞者效应': out['delta_contrib'][obs_idx],
    '线性特征': out['linear_contrib'][obs_idx],
    '样条特征': out['spline_contrib'][obs_idx],
}
plt.bar(contributions.keys(), contributions.values())
plt.title(f'观测{obs_idx}的μ分解')
```

### 4. 名人效应排名

```python
alpha = out['alpha']  # [n_celebs]
ranking = np.argsort(alpha)[::-1]
for i, celeb_id in enumerate(ranking[:10]):
    print(f"{i+1}. 名人{celeb_id}: α = {alpha[celeb_id]:.3f}")
```

