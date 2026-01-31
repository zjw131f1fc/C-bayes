# 编码规范

## 环境

conda activate bayes

## 题目理解（重要！）

### 第一问核心任务

**目标**：估算粉丝投票的分布，而不是预测淘汰结果

**题目原文**：
> - 构建数学模型，**估计每位选手在各周获得的粉丝投票数**
> - 你的模型是否能生成**与实际淘汰结果一致**的粉丝投票？
> - 请给出一致性度量指标

**评估指标**：
- 一致性：生成的粉丝投票能否复现真实淘汰结果（目标99%+）
- 不确定性：粉丝投票估计的置信区间宽度

### 我们的方案：贝叶斯模型 + 后验筛选

1. **贝叶斯模型**：建模 P_fan = f(名人特征, 舞者效应, ...)
2. **后验筛选**：只保留能复现淘汰结果的后验样本
3. **输出**：筛选后样本的 P_fan 分布（均值 + 置信区间）

**优势**：
- 保留特征建模能力（可回答第三问：特征影响分析）
- 满足一致性要求（筛选后复现率接近100%）
- 可量化不确定性

### 当前状态

- 原始后验样本复现率：53.6%（264周平均）
- 复现率=0%的周：39周（可能是"争议性"周）
- 需要实现后验筛选逻辑

## 项目概述

MCM 2026 Problem C - 与星共舞(Dancing with the Stars)粉丝投票预测
使用分层贝叶斯模型 + GAM 框架，基于 NumPyro/JAX 实现

## 目录结构

```
C-bayes/
├── main.py                 # 主入口：训练、评估、交叉验证
├── config.yaml             # 模型配置（特征、先验、采样参数）
├── src/
│   ├── model.py            # 贝叶斯模型定义（NumPyro）
│   ├── preprocess.py       # 数据加载、校验、过滤
│   ├── visualize.py        # 诊断可视化
│   └── utils.py            # 工具函数（load_config, save_data）
├── problem1_src/           # 第一问专用代码
│   └── visualize.py        # 第一问可视化
├── problem2_src/           # 第二问专用代码
│   ├── fpi.py              # FPI计算
│   ├── controversial.py    # 争议性分析
│   └── visualize.py        # 第二问可视化
├── data/
│   └── datas.pkl           # 预处理后的数据
├── outputs/
│   ├── model/              # 模型输出
│   └── results/            # 结果文件
├── visualize_problem1.py   # 第一问可视化入口
├── visualize_problem2.py   # 第二问可视化入口
└── docs/                   # 文档
    ├── 数据接口规范.md
    ├── MODEL_OUTPUT.md
    └── 特征.md
```

## 核心接口

```python
config = load_config(path)           # 加载YAML配置
datas = load_data(path)              # 加载数据字典
save_data(datas, path)               # 保存数据字典
```

## 统一函数签名

```python
datas = process_xxx(config, datas)   # 数据处理，链式更新datas
model = train(config, model, datas)  # 模型训练
datas = generate_xxx(config, model, datas)  # 模型推理
visualize_xxx(config, datas, tag)    # 可视化，输出到 outputs/{tag}/
```

## 数据结构 (datas['train_data'])

```python
{
    # 维度
    'n_obs': int,           # 观测数
    'n_weeks': int,         # 周数
    'n_celebs': int,        # 名人数
    'n_pros': int,          # 舞者数
    'n_seasons': int,       # 赛季数

    # 索引 (int32, shape=[n_obs])
    'celeb_idx': np.ndarray,   # 名人索引
    'pro_idx': np.ndarray,     # 舞者索引
    'week_idx': np.ndarray,    # 周索引
    'season_idx': np.ndarray,  # 赛季索引

    # 特征矩阵 (float32)
    'X_celeb': np.ndarray,     # [n_celebs, n_celeb_features]
    'X_pro': np.ndarray,       # [n_pros, n_pro_features]
    'X_obs': np.ndarray,       # [n_obs, n_obs_features]
    'X_celeb_names': list,     # 名人特征名
    'X_pro_names': list,       # 舞者特征名
    'X_obs_names': list,       # 观测特征名

    # 评委分数据 (float32, shape=[n_obs])
    'judge_score_pct': np.ndarray,   # 百分比法评委分
    'judge_rank_score': np.ndarray,  # 排名法评委分

    # 周级数据
    'week_data': [
        {
            'obs_mask': np.ndarray,       # bool, [n_obs]
            'eliminated_mask': np.ndarray, # bool, [n_obs]
            'n_contestants': int,
            'n_eliminated': int,
            'rule_method': int,           # 1=百分比法, 0=排名法
            'judge_save_active': bool,    # 评委拯救是否激活
            'season': int,
            'week': int,
        },
        ...
    ]
}
```

## 模型核心流程 (main.py)

```python
config = load_config('config.yaml')
datas = {}

datas = load_data('data/datas.pkl', datas)
datas = validate_data(datas)

model = build_model(config, datas)
datas = train(config, model, datas)
datas = extract_posterior(config, datas)
datas = predict(config, datas, datas)
datas = compute_metrics(config, datas)
datas = generate_output(config, datas)

visualize_diagnostics(config, datas, datas)
save_data(datas, 'outputs/results/results.pkl')
```

## 配置文件关键项 (config.yaml)

```yaml
model:
  spline_features: []              # 观测级样条特征
  celeb_spline_features: [age]     # 名人级样条特征
  exclude_obs_features: [...]      # 排除的观测特征
  exclude_celeb_features: [...]    # 排除的名人特征
  obs_interaction_features: [...]  # 观测特征交互项
  positive_prior_features: [...]   # 正向先验特征
  use_horseshoe: false             # Horseshoe正则化
  p_fan_normalize: softmax         # P_fan归一化方式
  use_judge_save_correction: true  # 评委拯救修正

prior:
  sigma_celeb: {a: 2.0, b: 1.0}    # 名人效应方差先验
  beta_obs_scale: 1.0              # 观测特征系数先验尺度

sampling:
  n_chains: 4
  n_tune: 500
  n_samples: 1000
  target_accept: 0.85
```

## 关键函数说明

### src/model.py
- `build_model(config, datas)`: 构建模型参数，处理特征交互项和样条
- `train(config, model_params, datas)`: MCMC采样
- `predict(config, train_datas, eval_datas)`: 后验预测
- `compute_metrics(config, datas)`: 计算准确率指标
- `generate_output(config, datas)`: 生成模型输出

### src/preprocess.py
- `load_data(path, datas)`: 加载pickle数据
- `validate_data(datas)`: 校验数据格式
- `filter_data(datas, test_season)`: 按赛季划分训练/测试集

## 运行命令

```bash
# 全量训练
python main.py

# 第一问可视化
python visualize_problem1.py

# 第二问可视化
python visualize_problem2.py
```
