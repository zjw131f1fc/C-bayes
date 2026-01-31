"""分析特征共线性并生成配置建议"""

import pandas as pd
import numpy as np

def load_correlation_matrix(path):
    """加载相关系数矩阵"""
    df = pd.read_csv(path, index_col=0)
    # 清理列名和数据
    df.columns = [c.strip() for c in df.columns]
    df.index = [c.strip() for c in df.index]
    df = df.apply(lambda x: pd.to_numeric(x.astype(str).str.strip(), errors='coerce'))
    return df

def find_high_correlations(corr_matrix, threshold=0.7):
    """找出高度相关的特征对"""
    high_corr = []
    features = corr_matrix.columns.tolist()

    for i, f1 in enumerate(features):
        for j, f2 in enumerate(features):
            if i < j:  # 只看上三角
                r = corr_matrix.loc[f1, f2]
                if abs(r) >= threshold:
                    high_corr.append((f1, f2, r))

    # 按相关系数绝对值排序
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    return high_corr

def suggest_features_to_remove(high_corr_pairs, keep_priority=None):
    """
    建议移除的特征
    keep_priority: 优先保留的特征列表
    """
    if keep_priority is None:
        keep_priority = []

    to_remove = set()
    kept = set()

    for f1, f2, r in high_corr_pairs:
        if f1 in to_remove or f2 in to_remove:
            continue  # 已经移除了其中一个

        # 决定移除哪个
        f1_priority = f1 in keep_priority
        f2_priority = f2 in keep_priority

        if f1_priority and not f2_priority:
            to_remove.add(f2)
            kept.add(f1)
        elif f2_priority and not f1_priority:
            to_remove.add(f1)
            kept.add(f2)
        else:
            # 都有优先级或都没有，移除第二个
            to_remove.add(f2)
            kept.add(f1)

    return to_remove, kept

def main():
    print("=" * 70)
    print("特征共线性分析")
    print("=" * 70)

    # 加载相关系数矩阵
    corr = load_correlation_matrix('feature_correlation_matrix.csv')
    print(f"\n特征数量: {len(corr.columns)}")

    # 找出高度相关的特征对
    print("\n" + "=" * 70)
    print("高度相关的特征对 (|r| >= 0.7)")
    print("=" * 70)
    high_corr = find_high_correlations(corr, threshold=0.7)
    for f1, f2, r in high_corr:
        print(f"  {f1:40s} <-> {f2:40s}: r={r:.3f}")

    print("\n" + "=" * 70)
    print("中等相关的特征对 (0.5 <= |r| < 0.7)")
    print("=" * 70)
    mid_corr = find_high_correlations(corr, threshold=0.5)
    mid_corr = [(f1, f2, r) for f1, f2, r in mid_corr if abs(r) < 0.7]
    for f1, f2, r in mid_corr:
        print(f"  {f1:40s} <-> {f2:40s}: r={r:.3f}")

    # 优先保留的特征（基于预测能力和可解释性）
    keep_priority = [
        'z_score',           # 最强预测因子
        'age',               # 基础特征
        'is_male',           # 基础特征
        'weeks_survived',    # 重要时序特征
        'times_in_bottom',   # 重要历史特征
        'teflon_factor',     # 重要特征
        'season_era',        # 时代效应
        'pro_prev_wins',     # 舞者能力
    ]

    print("\n" + "=" * 70)
    print("建议移除的特征 (解决高度共线性)")
    print("=" * 70)

    to_remove, kept = suggest_features_to_remove(high_corr, keep_priority)

    print("\n保留的特征:")
    for f in sorted(kept):
        print(f"  ✓ {f}")

    print("\n建议移除的特征:")
    for f in sorted(to_remove):
        print(f"  ✗ {f}")

    # 生成配置
    print("\n" + "=" * 70)
    print("配置建议 (config.yaml)")
    print("=" * 70)
    print("\nexclude_features:")
    for f in sorted(to_remove):
        print(f"  - {f}")

    # 分析 z_score 相关的特征
    print("\n" + "=" * 70)
    print("z_score 相关特征分析")
    print("=" * 70)
    z_corr = corr.loc['z_score'].sort_values(key=abs, ascending=False)
    print("\nz_score 与其他特征的相关性 (|r| > 0.2):")
    for feat, r in z_corr.items():
        if feat != 'z_score' and abs(r) > 0.2:
            print(f"  {feat:40s}: r={r:.3f}")

    return to_remove

if __name__ == "__main__":
    to_remove = main()
