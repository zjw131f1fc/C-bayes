"""
BART交互检测脚本（处理共线性后）
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# 设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(path):
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded['train_data']


def prepare_features_for_bart(td):
    """
    准备BART特征，处理共线性问题
    """
    n_obs = td['n_obs']
    celeb_idx = td['celeb_idx']

    features = {}
    feature_groups = {}  # 记录哪些特征是等价的

    # ========== 名人特征 ==========
    X_celeb = td['X_celeb']
    celeb_names = td['X_celeb_names']

    for i, name in enumerate(celeb_names):
        # 跳过冗余特征
        if name == 'log_us_state_pop':
            # 与 is_international 高度相关，保留 is_international
            feature_groups['is_international'] = ['is_international', 'log_us_state_pop']
            continue

        # 行业变量：删除一个作为基准（避免完美共线性）
        if name == 'industry_Performing Arts & Entertainment':
            # 作为基准类别删除
            continue

        features[name] = X_celeb[celeb_idx, i]

    # ========== 观测特征 ==========
    X_obs = td['X_obs']
    obs_names = td['X_obs_names']

    for i, name in enumerate(obs_names):
        # 跳过冗余特征
        if name == 'youtube_comment_count_norm':
            # 与 view_count 高度相关
            feature_groups['youtube_view_count_norm'] = ['youtube_view_count_norm', 'youtube_comment_count_norm']
            continue

        if name == 'pro_is_male':
            # 与 is_male 高度相关（异性配对）
            feature_groups['is_male'] = ['is_male', 'pro_is_male']
            continue

        features[name] = X_obs[:, i]

    # ========== 评委分 ==========
    # 只保留一个评委分（两者高度相关）
    features['judge_score_pct'] = td['judge_score_pct']
    feature_groups['judge_score_pct'] = ['judge_score_pct', 'judge_rank_score']

    # 构建特征矩阵
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])

    # 构建目标变量（淘汰标记）
    y = np.zeros(n_obs, dtype=np.float32)
    for wd in td['week_data']:
        y[wd['eliminated_mask']] = 1.0

    print(f"处理后特征数: {len(feature_names)} (原始: {len(celeb_names) + len(obs_names) + 2})")
    print(f"\n删除的冗余特征:")
    print(f"  - log_us_state_pop (与 is_international 相关 r=-0.99)")
    print(f"  - youtube_comment_count_norm (与 view_count 相关 r=0.96)")
    print(f"  - pro_is_male (与 is_male 相关 r=-0.89)")
    print(f"  - judge_rank_score (与 judge_score_pct 相关 r=0.78)")
    print(f"  - industry_Performing Arts & Entertainment (作为基准类别)")

    return X, y, feature_names, feature_groups


def run_bart_analysis(X, y, feature_names, n_trees=50, n_samples=500):
    """
    运行BART并提取变量重要性和交互
    """
    try:
        import pymc as pm
        import pymc_bart as pmb
        import arviz as az
    except ImportError:
        print("需要安装 pymc-bart: pip install pymc-bart")
        return None, None

    print(f"\n运行BART: {n_trees} trees, {n_samples} samples")
    print("这可能需要几分钟...")

    with pm.Model() as model:
        # BART模型
        mu = pmb.BART('mu', X=X, Y=y, m=n_trees)

        # 伯努利似然（淘汰是二分类）
        p = pm.Deterministic('p', pm.math.sigmoid(mu))
        likelihood = pm.Bernoulli('y', p=p, observed=y)

        # 采样
        trace = pm.sample(n_samples, tune=200, cores=2, random_seed=42,
                         return_inferencedata=True)

    # 提取变量重要性
    var_importance = pmb.compute_variable_importance(trace, X=X, m=n_trees)

    return trace, var_importance


def analyze_interactions_simple(X, y, feature_names, top_k=10):
    """
    简单的交互检测：基于梯度提升树的特征重要性
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        import itertools
    except ImportError:
        print("需要安装 scikit-learn")
        return None

    print("\n使用梯度提升树检测交互...")

    # 训练基础模型
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
    gb.fit(X, y)

    # 特征重要性
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gb.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n单特征重要性 (Top 10):")
    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']:40s} {row['importance']:.4f}")

    # 检测交互：添加交互项后的重要性提升
    print("\n检测二阶交互...")
    top_features = importance.head(top_k)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features]

    interactions = []
    for i, j in itertools.combinations(range(len(top_indices)), 2):
        idx_i, idx_j = top_indices[i], top_indices[j]
        name_i, name_j = top_features[i], top_features[j]

        # 添加交互项
        interaction_term = X[:, idx_i] * X[:, idx_j]
        X_with_interaction = np.column_stack([X, interaction_term])

        # 重新训练
        gb_inter = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        gb_inter.fit(X_with_interaction, y)

        # 交互项的重要性
        inter_importance = gb_inter.feature_importances_[-1]
        interactions.append({
            'feature_1': name_i,
            'feature_2': name_j,
            'interaction_importance': inter_importance
        })

    interactions_df = pd.DataFrame(interactions).sort_values('interaction_importance', ascending=False)

    print("\n二阶交互重要性 (Top 15):")
    for _, row in interactions_df.head(15).iterrows():
        print(f"  {row['feature_1']:25s} × {row['feature_2']:25s} {row['interaction_importance']:.4f}")

    return importance, interactions_df


def main():
    print("=" * 60)
    print("BART交互检测（处理共线性后）")
    print("=" * 60)

    # 加载数据
    td = load_data('data/datas.pkl')

    # 准备特征
    X, y, feature_names, feature_groups = prepare_features_for_bart(td)

    # 创建输出目录
    output_dir = Path('outputs/bart_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 简单交互检测（快速）
    importance, interactions = analyze_interactions_simple(X, y, feature_names)

    if importance is not None:
        # 保存结果
        importance.to_csv(output_dir / 'feature_importance.csv', index=False)
        interactions.to_csv(output_dir / 'interactions.csv', index=False)

        # 可视化
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 特征重要性
        top_imp = importance.head(15)
        axes[0].barh(range(len(top_imp)), top_imp['importance'].values)
        axes[0].set_yticks(range(len(top_imp)))
        axes[0].set_yticklabels(top_imp['feature'].values)
        axes[0].invert_yaxis()
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Feature Importance (Top 15)')

        # 交互重要性
        top_inter = interactions.head(15)
        labels = [f"{r['feature_1'][:12]} × {r['feature_2'][:12]}" for _, r in top_inter.iterrows()]
        axes[1].barh(range(len(top_inter)), top_inter['interaction_importance'].values)
        axes[1].set_yticks(range(len(top_inter)))
        axes[1].set_yticklabels(labels)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Interaction Importance')
        axes[1].set_title('Interaction Importance (Top 15)')

        plt.tight_layout()
        plt.savefig(output_dir / 'importance_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n结果已保存到: {output_dir}/")

    # 提示等价特征组
    print("\n" + "=" * 60)
    print("注意：以下特征组是等价的（高度相关）")
    print("=" * 60)
    for main_feat, group in feature_groups.items():
        print(f"  {main_feat} ≈ {group[1:]}")
    print("\n如果发现某个交互重要，其等价特征的交互也可能重要")

    # 建议
    print("\n" + "=" * 60)
    print("建议添加到贝叶斯模型的交互项")
    print("=" * 60)
    if interactions is not None:
        top_interactions = interactions.head(5)
        print("\n在 config.yaml 中添加:")
        print("model:")
        print("  obs_interaction_features:")
        for _, row in top_interactions.iterrows():
            print(f"    - [{row['feature_1']}, {row['feature_2']}]")


if __name__ == '__main__':
    main()
