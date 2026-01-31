"""简单线性模型 - 快速验证特征有效性
目标函数：回归预测"受欢迎程度"分数，被淘汰者分数低
评估方式：每周内按分数排序，选择最低的作为预测淘汰者
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, load_data


def build_feature_matrix(config, train_data):
    """构建特征矩阵，与贝叶斯模型保持一致"""
    model_cfg = config['model']

    # 获取排除的特征
    exclude_obs = set(model_cfg.get('exclude_obs_features', []))
    exclude_celeb = set(model_cfg.get('exclude_celeb_features', []))

    # 观测级特征
    X_obs = train_data['X_obs']
    obs_names = train_data['X_obs_names']
    obs_mask = [i for i, name in enumerate(obs_names) if name not in exclude_obs]
    X_obs_filtered = X_obs[:, obs_mask]
    obs_names_filtered = [obs_names[i] for i in obs_mask]

    # 名人级特征 -> 扩展到观测级
    X_celeb = train_data['X_celeb']
    celeb_names = train_data['X_celeb_names']
    celeb_mask = [i for i, name in enumerate(celeb_names) if name not in exclude_celeb]
    X_celeb_filtered = X_celeb[:, celeb_mask]
    celeb_names_filtered = [celeb_names[i] for i in celeb_mask]

    celeb_idx = train_data['celeb_idx']
    X_celeb_expanded = X_celeb_filtered[celeb_idx]

    # 交互项
    interaction_features = []
    interaction_names = []
    for expr in model_cfg.get('obs_interaction_features', []) or []:
        parts = expr.replace(' ', '').split('*')
        if len(parts) == 2:
            name1, name2 = parts
            idx1 = obs_names.index(name1) if name1 in obs_names else None
            idx2 = obs_names.index(name2) if name2 in obs_names else None
            if idx1 is not None and idx2 is not None:
                interaction_features.append(X_obs[:, idx1] * X_obs[:, idx2])
                interaction_names.append(f"{name1}*{name2}")

    if interaction_features:
        X_interaction = np.column_stack(interaction_features)
    else:
        X_interaction = np.zeros((X_obs.shape[0], 0))

    # 合并所有特征
    X = np.hstack([X_obs_filtered, X_celeb_expanded, X_interaction])
    feature_names = obs_names_filtered + [f"celeb_{n}" for n in celeb_names_filtered] + interaction_names

    return X, feature_names


def build_labels(train_data):
    """构建标签：未淘汰=1，被淘汰=0（分数越高越受欢迎）"""
    week_data = train_data['week_data']
    n_obs = train_data['n_obs']

    y = np.ones(n_obs, dtype=np.float32)  # 默认未淘汰=1
    week_idx = np.zeros(n_obs, dtype=np.int32)

    for w, wd in enumerate(week_data):
        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        week_idx[obs_mask] = w
        y[elim_mask] = 0  # 被淘汰=0

    return y, week_idx


def evaluate_per_week(model, X, train_data):
    """按周评估淘汰预测准确率"""
    week_data = train_data['week_data']
    scores = model.predict(X)  # 预测受欢迎程度分数

    correct = 0
    total = 0

    for w, wd in enumerate(week_data):
        if wd['n_eliminated'] == 0:
            continue

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']

        # 获取该周的预测分数
        week_scores = scores[obs_mask]

        # 预测：选择分数最低的 n_elim 个
        pred_elim_idx = np.argsort(week_scores)[:n_elim]

        # 真实淘汰者在该周内的索引
        week_obs_indices = np.where(obs_mask)[0]
        true_elim_global = np.where(elim_mask)[0]
        true_elim_idx = [np.where(week_obs_indices == g)[0][0] for g in true_elim_global]

        # 计算准确率
        correct += len(set(pred_elim_idx) & set(true_elim_idx))
        total += n_elim

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("线性模型特征验证")
    print("=" * 60)

    # 加载配置和数据
    config = load_config('config.yaml')
    datas = load_data('data/datas.pkl')
    train_data = datas['train_data']

    # 构建特征和标签
    X, feature_names = build_feature_matrix(config, train_data)
    y, week_idx = build_labels(train_data)

    print(f"\n数据规模:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  淘汰样本: {int((y==0).sum())} ({(y==0).mean()*100:.1f}%)")

    # 检查特征
    print("\n" + "-" * 60)
    print("特征检查")
    print("-" * 60)
    print(f"{'特征名':45s} {'均值':>10} {'标准差':>10} {'最小':>10} {'最大':>10}")
    for i, name in enumerate(feature_names):
        col = X[:, i]
        print(f"{name:45s} {col.mean():>10.3f} {col.std():>10.3f} {col.min():>10.3f} {col.max():>10.3f}")

    # 检查特征与淘汰的相关性
    print("\n" + "-" * 60)
    print("特征与淘汰的相关性 (正=淘汰者该特征更高)")
    print("-" * 60)
    for i, name in enumerate(feature_names):
        col = X[:, i]
        # 淘汰者 vs 未淘汰者的均值差
        elim_mean = col[y == 0].mean()
        surv_mean = col[y == 1].mean()
        diff = elim_mean - surv_mean
        corr = np.corrcoef(col, 1-y)[0, 1]  # 与淘汰的相关系数
        print(f"{name:45s} 淘汰均值={elim_mean:>7.3f} 存活均值={surv_mean:>7.3f} 差={diff:>7.3f} r={corr:>6.3f}")

    # 检查 cumulative_avg_score 的具体值
    print("\n" + "-" * 60)
    print("cumulative_avg_score 详细检查")
    print("-" * 60)

    # 找到这个特征的索引
    cum_idx = feature_names.index('cumulative_avg_score') if 'cumulative_avg_score' in feature_names else None
    judge_idx = feature_names.index('judge_score_season_zscore') if 'judge_score_season_zscore' in feature_names else None

    if cum_idx is not None:
        week_data = train_data['week_data']
        celeb_idx_arr = train_data['celeb_idx']

        # 打印前几周每个选手的 cumulative_avg_score
        for w in range(min(5, len(week_data))):
            wd = week_data[w]
            obs_mask = wd['obs_mask']
            elim_mask = wd['eliminated_mask']

            week_obs = np.where(obs_mask)[0]
            print(f"\n第 {w+1} 周:")
            print(f"  {'选手ID':>8} {'cum_avg':>10} {'judge_z':>10} {'淘汰':>6}")

            for obs_i in week_obs:
                cum_val = X[obs_i, cum_idx]
                judge_val = X[obs_i, judge_idx] if judge_idx else 0
                is_elim = "是" if elim_mask[obs_i] else ""
                celeb_id = celeb_idx_arr[obs_i]
                print(f"  {celeb_id:>8} {cum_val:>10.3f} {judge_val:>10.3f} {is_elim:>6}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 岭回归
    print("\n" + "-" * 60)
    print("岭回归 (不同正则化强度)")
    print("-" * 60)

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_scaled, y)
        acc = evaluate_per_week(ridge, X_scaled, train_data)
        print(f"  alpha={alpha:6.2f}: 淘汰预测准确率={acc:.3f}")

    # 最佳模型的特征系数
    print("\n" + "-" * 60)
    print("特征系数 (alpha=1.0)")
    print("-" * 60)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    coef_idx = np.argsort(np.abs(ridge.coef_))[::-1]
    for i in coef_idx[:15]:
        sign = "+" if ridge.coef_[i] > 0 else "-"
        print(f"  {feature_names[i]:45s} {sign}{abs(ridge.coef_[i]):.4f}")

    print("\n" + "=" * 60)

    # 打印几周的详细数据用于检查
    print("\n详细数据检查 (前3周有淘汰的)")
    print("-" * 60)

    week_data = train_data['week_data']
    scores = ridge.predict(X_scaled)
    checked = 0

    for w, wd in enumerate(week_data):
        if wd['n_eliminated'] == 0:
            continue
        if checked >= 3:
            break

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']

        week_obs_indices = np.where(obs_mask)[0]
        week_scores = scores[obs_mask]
        week_y = y[obs_mask]

        # 获取名人名字（如果有的话）
        celeb_idx_week = train_data['celeb_idx'][obs_mask]

        print(f"\n第 {w+1} 周 (淘汰 {n_elim} 人):")
        print(f"  {'选手ID':>8} {'预测分':>10} {'实际':>6} {'judge_zscore':>12}")

        # 按预测分排序
        sort_idx = np.argsort(week_scores)
        for i in sort_idx:
            global_idx = week_obs_indices[i]
            status = "淘汰" if week_y[i] == 0 else ""
            judge_z = X[global_idx, feature_names.index('judge_score_season_zscore')] if 'judge_score_season_zscore' in feature_names else 0
            print(f"  {celeb_idx_week[i]:>8} {week_scores[i]:>10.4f} {status:>6} {judge_z:>12.3f}")

        # 预测结果
        pred_elim_idx = sort_idx[:n_elim]
        true_elim_idx = np.where(week_y == 0)[0]
        hit = len(set(pred_elim_idx) & set(true_elim_idx))
        print(f"  预测淘汰: {list(celeb_idx_week[pred_elim_idx])}")
        print(f"  实际淘汰: {list(celeb_idx_week[true_elim_idx])}")
        print(f"  命中: {hit}/{n_elim}")

        checked += 1


if __name__ == '__main__':
    main()
