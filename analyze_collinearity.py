"""
多重共线性分析脚本
分析特征之间的相关性、VIF、以及对BART交互检测的影响
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data(path):
    """加载数据"""
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    return loaded['train_data']


def compute_vif(X, feature_names):
    """计算方差膨胀因子 (VIF)"""
    from numpy.linalg import lstsq

    n_features = X.shape[1]
    vif_values = []

    for i in range(n_features):
        # 用其他特征预测第i个特征
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)

        # 添加截距
        X_with_intercept = np.column_stack([np.ones(X_others.shape[0]), X_others])

        # 最小二乘拟合
        try:
            coeffs, residuals, rank, s = lstsq(X_with_intercept, y, rcond=None)
            y_pred = X_with_intercept @ coeffs

            # 计算 R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)

            if ss_tot == 0:
                r_squared = 0
            else:
                r_squared = 1 - ss_res / ss_tot

            # VIF = 1 / (1 - R²)
            if r_squared >= 1:
                vif = np.inf
            else:
                vif = 1 / (1 - r_squared)
        except:
            vif = np.nan

        vif_values.append(vif)

    return pd.DataFrame({
        'feature': feature_names,
        'VIF': vif_values
    }).sort_values('VIF', ascending=False)


def analyze_correlation(X, feature_names, title, output_path):
    """分析相关性矩阵"""
    df = pd.DataFrame(X, columns=feature_names)
    corr = df.corr()

    # 找出高相关对
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = corr.iloc[i, j]
            if abs(r) > 0.5:
                high_corr_pairs.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'correlation': r
                })

    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)

    # 绘制热力图
    fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.8), max(8, len(feature_names) * 0.6)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, ax=ax,
                annot_kws={'size': 8})
    ax.set_title(f'{title} - Correlation Matrix')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return corr, high_corr_df


def analyze_combined_features(td):
    """分析组合后的特征（观测级展开）"""
    # 构建观测级完整特征矩阵
    n_obs = td['n_obs']
    celeb_idx = td['celeb_idx']

    # 名人特征展开到观测级
    X_celeb_expanded = td['X_celeb'][celeb_idx]

    # 合并所有特征
    X_combined = np.hstack([X_celeb_expanded, td['X_obs']])
    combined_names = td['X_celeb_names'] + td['X_obs_names']

    # 添加评委分
    X_combined = np.hstack([X_combined,
                           td['judge_score_pct'].reshape(-1, 1),
                           td['judge_rank_score'].reshape(-1, 1)])
    combined_names = combined_names + ['judge_score_pct', 'judge_rank_score']

    return X_combined, combined_names


def main():
    print("=" * 60)
    print("多重共线性分析")
    print("=" * 60)

    # 加载数据
    td = load_data('data/datas.pkl')
    print(f"\n数据维度: {td['n_obs']} obs, {td['n_celebs']} celebs, {td['n_pros']} pros")

    # 创建输出目录
    output_dir = Path('outputs/collinearity')
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 名人特征分析 ==========
    print("\n" + "=" * 40)
    print("1. 名人特征 (X_celeb) 分析")
    print("=" * 40)

    print(f"\n特征列表 ({len(td['X_celeb_names'])} 个):")
    for i, name in enumerate(td['X_celeb_names']):
        print(f"  {i+1}. {name}")

    corr_celeb, high_corr_celeb = analyze_correlation(
        td['X_celeb'], td['X_celeb_names'],
        'Celebrity Features', output_dir / 'corr_celeb.png'
    )

    print(f"\n高相关特征对 (|r| > 0.5):")
    if len(high_corr_celeb) > 0:
        for _, row in high_corr_celeb.iterrows():
            print(f"  {row['feature_1']} <-> {row['feature_2']}: r = {row['correlation']:.3f}")
    else:
        print("  无")

    # VIF分析
    if td['X_celeb'].shape[1] > 1:
        vif_celeb = compute_vif(td['X_celeb'], td['X_celeb_names'])
        print(f"\nVIF (方差膨胀因子):")
        print("  VIF > 10: 严重共线性")
        print("  VIF > 5: 中等共线性")
        print("  VIF < 5: 可接受")
        print()
        for _, row in vif_celeb.iterrows():
            status = "⚠️ 严重" if row['VIF'] > 10 else ("⚠️ 中等" if row['VIF'] > 5 else "✓")
            print(f"  {row['feature']:30s} VIF = {row['VIF']:8.2f}  {status}")

    # ========== 2. 观测特征分析 ==========
    print("\n" + "=" * 40)
    print("2. 观测特征 (X_obs) 分析")
    print("=" * 40)

    print(f"\n特征列表 ({len(td['X_obs_names'])} 个):")
    for i, name in enumerate(td['X_obs_names']):
        print(f"  {i+1}. {name}")

    corr_obs, high_corr_obs = analyze_correlation(
        td['X_obs'], td['X_obs_names'],
        'Observation Features', output_dir / 'corr_obs.png'
    )

    print(f"\n高相关特征对 (|r| > 0.5):")
    if len(high_corr_obs) > 0:
        for _, row in high_corr_obs.iterrows():
            print(f"  {row['feature_1']} <-> {row['feature_2']}: r = {row['correlation']:.3f}")
    else:
        print("  无")

    # VIF分析
    if td['X_obs'].shape[1] > 1:
        vif_obs = compute_vif(td['X_obs'], td['X_obs_names'])
        print(f"\nVIF (方差膨胀因子):")
        for _, row in vif_obs.iterrows():
            status = "⚠️ 严重" if row['VIF'] > 10 else ("⚠️ 中等" if row['VIF'] > 5 else "✓")
            print(f"  {row['feature']:30s} VIF = {row['VIF']:8.2f}  {status}")

    # ========== 3. 组合特征分析 ==========
    print("\n" + "=" * 40)
    print("3. 组合特征分析 (名人+观测+评委分)")
    print("=" * 40)

    X_combined, combined_names = analyze_combined_features(td)
    print(f"\n组合后特征数: {len(combined_names)}")

    corr_combined, high_corr_combined = analyze_correlation(
        X_combined, combined_names,
        'Combined Features', output_dir / 'corr_combined.png'
    )

    print(f"\n高相关特征对 (|r| > 0.5):")
    if len(high_corr_combined) > 0:
        for _, row in high_corr_combined.head(20).iterrows():
            print(f"  {row['feature_1']} <-> {row['feature_2']}: r = {row['correlation']:.3f}")
        if len(high_corr_combined) > 20:
            print(f"  ... 还有 {len(high_corr_combined) - 20} 对")
    else:
        print("  无")

    # 组合特征VIF
    if X_combined.shape[1] > 1:
        vif_combined = compute_vif(X_combined, combined_names)
        print(f"\nVIF (方差膨胀因子) - Top 15:")
        for _, row in vif_combined.head(15).iterrows():
            status = "⚠️ 严重" if row['VIF'] > 10 else ("⚠️ 中等" if row['VIF'] > 5 else "✓")
            print(f"  {row['feature']:30s} VIF = {row['VIF']:8.2f}  {status}")

    # ========== 4. 评委分相关性 ==========
    print("\n" + "=" * 40)
    print("4. 评委分相关性分析")
    print("=" * 40)

    judge_corr = np.corrcoef(td['judge_score_pct'], td['judge_rank_score'])[0, 1]
    print(f"\njudge_score_pct vs judge_rank_score: r = {judge_corr:.3f}")
    if abs(judge_corr) > 0.8:
        print("  ⚠️ 高度相关，建议只保留一个")
    elif abs(judge_corr) > 0.5:
        print("  ⚠️ 中等相关，BART可能会分散重要性")
    else:
        print("  ✓ 相关性可接受")

    # ========== 5. 总结与建议 ==========
    print("\n" + "=" * 60)
    print("总结与建议")
    print("=" * 60)

    # 收集问题
    problems = []

    # 检查高VIF
    if td['X_celeb'].shape[1] > 1:
        high_vif_celeb = vif_celeb[vif_celeb['VIF'] > 5]['feature'].tolist()
        if high_vif_celeb:
            problems.append(f"名人特征高VIF: {high_vif_celeb}")

    if td['X_obs'].shape[1] > 1:
        high_vif_obs = vif_obs[vif_obs['VIF'] > 5]['feature'].tolist()
        if high_vif_obs:
            problems.append(f"观测特征高VIF: {high_vif_obs}")

    # 检查高相关
    if len(high_corr_celeb) > 0:
        problems.append(f"名人特征高相关对: {len(high_corr_celeb)} 对")
    if len(high_corr_obs) > 0:
        problems.append(f"观测特征高相关对: {len(high_corr_obs)} 对")

    if problems:
        print("\n发现的共线性问题:")
        for p in problems:
            print(f"  - {p}")

        print("\n对BART交互检测的影响:")
        print("  1. 变量重要性会在相关变量间分散")
        print("  2. 可能检测到伪交互（实际是共线性）")
        print("  3. 真实交互可能被低估")

        print("\n建议处理方案:")
        print("  A) 删除冗余变量（保留业务意义更强的）")
        print("  B) 对高相关变量组做PCA")
        print("  C) 使用正则化（Horseshoe/LASSO）让模型自动选择")
        print("  D) 分析时注意：相关变量的交互可能是等价的")
    else:
        print("\n✓ 未发现严重的共线性问题，可以直接使用BART")

    print(f"\n相关性热力图已保存到: {output_dir}/")

    # 保存详细结果
    results = {
        'corr_celeb': corr_celeb,
        'corr_obs': corr_obs,
        'corr_combined': corr_combined,
        'high_corr_celeb': high_corr_celeb,
        'high_corr_obs': high_corr_obs,
        'high_corr_combined': high_corr_combined,
    }
    if td['X_celeb'].shape[1] > 1:
        results['vif_celeb'] = vif_celeb
    if td['X_obs'].shape[1] > 1:
        results['vif_obs'] = vif_obs
    if X_combined.shape[1] > 1:
        results['vif_combined'] = vif_combined

    with open(output_dir / 'collinearity_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"详细结果已保存到: {output_dir}/collinearity_results.pkl")


if __name__ == '__main__':
    main()
