"""诊断可视化模块"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from patsy import dmatrix


def _build_spline_basis(x, n_knots, degree):
    """构建B样条基矩阵"""
    knots = np.linspace(x.min(), x.max(), n_knots)
    basis = dmatrix(
        f"bs(x, knots={list(knots[1:-1])}, degree={degree}, include_intercept=False) - 1",
        {"x": x}, return_type='dataframe'
    )
    return np.asarray(basis, dtype=np.float32)


def visualize_diagnostics(config, train_datas, test_datas, fold_idx=None, output_dir='outputs/diagnostics'):
    """诊断可视化：分析线性/非线性关系

    Args:
        fold_idx: 折索引，None 表示全量训练（文件名不加 fold 标识）
    """
    from src.model import _parse_interaction

    os.makedirs(output_dir, exist_ok=True)

    # 文件名前缀
    prefix = f'fold{fold_idx}_' if fold_idx is not None else ''

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    posterior = train_datas['posterior_samples']
    td_train = train_datas['train_data']
    td_test = test_datas['train_data']
    model_cfg = config['model']

    # 获取观测特征配置
    X_obs_names_orig = list(td_train['X_obs_names'])
    X_obs_orig = td_train['X_obs'].copy()
    X_obs_names = list(X_obs_names_orig)
    X_obs_train = X_obs_orig.copy()
    spline_features = model_cfg['spline_features'] or []
    exclude_obs = model_cfg.get('exclude_obs_features') or []
    obs_interaction = model_cfg.get('obs_interaction_features') or []
    center_features = model_cfg.get('center_features', False)

    # 添加观测特征交互项
    if obs_interaction:
        for expr in obs_interaction:
            new_col, new_name = _parse_interaction(expr, X_obs_orig, X_obs_names_orig)
            X_obs_train = np.column_stack([X_obs_train, new_col])
            X_obs_names.append(new_name)

    # 过滤掉排除的观测特征
    if exclude_obs:
        keep_cols = [i for i, name in enumerate(X_obs_names) if name not in exclude_obs]
        X_obs_train = X_obs_train[:, keep_cols]
        X_obs_names = [X_obs_names[i] for i in keep_cols]

    # 使用训练集的均值进行中心化
    feature_means = train_datas.get('feature_means')
    if center_features and feature_means is not None:
        X_obs_train = X_obs_train - feature_means

    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    # === 图1: 各部分贡献的分布 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 计算训练集上的各部分贡献
    alpha_mean = posterior['alpha'].mean(axis=0)
    delta_mean = posterior['delta'].mean(axis=0)

    alpha_contrib = alpha_mean[td_train['celeb_idx']]
    delta_contrib = delta_mean[td_train['pro_idx']]

    X_lin = X_obs_train[:, linear_cols] if linear_cols else None
    linear_contrib = np.zeros(td_train['n_obs'])
    if X_lin is not None and 'beta_obs' in posterior:
        beta_obs = posterior['beta_obs'].mean(axis=0)
        linear_contrib = X_lin @ beta_obs

    spline_contrib = np.zeros(td_train['n_obs'])
    for i, col in enumerate(spline_cols):
        key = f'spline_{i}_coef'
        if key in posterior:
            coef = posterior[key].mean(axis=0)
            basis = _build_spline_basis(X_obs_train[:, col],
                                        model_cfg['n_spline_knots'], model_cfg['spline_degree'])
            spline_contrib += basis @ coef

    # 绘制各部分贡献的直方图
    axes[0, 0].hist(alpha_contrib, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title(f'Alpha (Celebrity) Contrib\nstd={np.std(alpha_contrib):.3f}')
    axes[0, 0].set_xlabel('Contribution')

    axes[0, 1].hist(delta_contrib, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title(f'Delta (Pro) Contrib\nstd={np.std(delta_contrib):.3f}')
    axes[0, 1].set_xlabel('Contribution')

    axes[1, 0].hist(linear_contrib, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title(f'Linear Features Contrib\nstd={np.std(linear_contrib):.3f}')
    axes[1, 0].set_xlabel('Contribution')

    axes[1, 1].hist(spline_contrib, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title(f'Spline (Nonlinear) Contrib\nstd={np.std(spline_contrib):.3f}')
    axes[1, 1].set_xlabel('Contribution')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}contributions.png', dpi=150)
    plt.close()

    # === 图2: 样条特征的非线性关系 ===
    n_spline = len(spline_cols)
    if n_spline > 0:
        fig, axes = plt.subplots(1, n_spline, figsize=(6*n_spline, 5))
        if n_spline == 1:
            axes = [axes]

        for i, col in enumerate(spline_cols):
            feature_name = X_obs_names[col]
            x_vals = X_obs_train[:, col]

            key = f'spline_{i}_coef'
            if key in posterior:
                coef = posterior[key].mean(axis=0)
                basis = _build_spline_basis(x_vals, model_cfg['n_spline_knots'], model_cfg['spline_degree'])
                y_spline = basis @ coef

                sort_idx = np.argsort(x_vals)
                axes[i].scatter(x_vals, y_spline, alpha=0.3, s=10, label='Data')
                axes[i].plot(x_vals[sort_idx], y_spline[sort_idx], 'r-', lw=2, label='Spline fit')
                axes[i].set_xlabel(feature_name)
                axes[i].set_ylabel('Spline Contribution')
                axes[i].set_title(f'{feature_name} Nonlinear Effect')
                axes[i].legend()
                axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}spline_effects.png', dpi=150)
        plt.close()

    # === 图3: 特征与mu的关系（检测遗漏的非线性） ===
    mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib

    n_features = len(X_obs_names)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = np.atleast_2d(axes).flatten()

    for i, name in enumerate(X_obs_names):
        x_vals = X_obs_train[:, i]
        axes[i].scatter(x_vals, mu, alpha=0.3, s=10)
        axes[i].set_xlabel(name, fontsize=11)
        axes[i].set_ylabel('mu', fontsize=11)
        axes[i].set_title(f'{name} vs mu', fontsize=12)

        try:
            sort_idx = np.argsort(x_vals)
            window = max(len(x_vals) // 20, 5)
            smoothed = uniform_filter1d(mu[sort_idx].astype(float), size=window)
            axes[i].plot(x_vals[sort_idx], smoothed, 'r-', lw=2, alpha=0.7, label='Trend')
            axes[i].legend()
        except:
            pass

    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}feature_vs_mu.png', dpi=150)
    plt.close()

    # === 图4: 测试集预测残差分析 ===
    S_samples = test_datas['S_samples']
    S_mean = S_samples.mean(axis=0)

    week_data = td_test['week_data']
    residuals = []
    actual_ranks = []
    pred_ranks = []

    for wd in week_data:
        if wd['n_eliminated'] == 0:
            continue
        mask = wd['obs_mask']
        S_week = S_mean[mask]

        pred_rank = np.argsort(np.argsort(S_week))
        elim_mask = wd['eliminated_mask'][mask]
        actual_rank = np.zeros_like(pred_rank)
        actual_rank[elim_mask] = 1

        residuals.extend(pred_rank[elim_mask] - 0)
        pred_ranks.extend(pred_rank)
        actual_ranks.extend(actual_rank)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pred_ranks = np.array(pred_ranks)
    actual_ranks = np.array(actual_ranks)

    axes[0].hist(pred_ranks[actual_ranks == 1], bins=20, alpha=0.7, label='Eliminated', density=True)
    axes[0].hist(pred_ranks[actual_ranks == 0], bins=20, alpha=0.7, label='Survived', density=True)
    axes[0].set_xlabel('Predicted Rank (0=lowest)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Eliminated vs Survived Rank Distribution')
    axes[0].legend()

    axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Eliminated Pred Rank (ideal=0)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Elimination Residuals\nmean={np.mean(residuals):.2f}, median={np.median(residuals):.0f}')
    axes[1].axvline(x=0, color='red', linestyle='--', label='Ideal')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{prefix}residuals.png', dpi=150)
    plt.close()

    # === 图5: 线性特征的效应 ===
    if linear_cols:
        n_lin = len(linear_cols)
        n_cols_plot = min(3, n_lin)
        n_rows_plot = (n_lin + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5*n_cols_plot, 4*n_rows_plot))
        axes = np.atleast_2d(axes).flatten()

        if 'beta_obs' in posterior:
            beta_samples = posterior['beta_obs']
            for i, col in enumerate(linear_cols):
                name = X_obs_names[col]
                beta_i = beta_samples[:, i]
                axes[i].hist(beta_i, bins=30, alpha=0.7, edgecolor='black')
                axes[i].axvline(x=0, color='red', linestyle='--')
                axes[i].set_title(f'{name}\nb={beta_i.mean():.3f}+/-{beta_i.std():.3f}', fontsize=10)
                axes[i].set_xlabel('Coefficient')

        for i in range(n_lin, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}linear_effects.png', dpi=150)
        plt.close()

    # === 图6: 名人特征 vs alpha（检测非线性） ===
    X_celeb = td_train['X_celeb']
    X_celeb_names = td_train['X_celeb_names']
    n_celeb_features = len(X_celeb_names)

    if n_celeb_features > 0:
        n_cols_plot = min(3, n_celeb_features)
        n_rows_plot = (n_celeb_features + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6*n_cols_plot, 5*n_rows_plot))
        axes = np.atleast_2d(axes).flatten()

        for i, name in enumerate(X_celeb_names):
            x_vals = X_celeb[:, i]
            axes[i].scatter(x_vals, alpha_mean, alpha=0.5, s=20)
            axes[i].set_xlabel(name, fontsize=11)
            axes[i].set_ylabel('alpha', fontsize=11)
            axes[i].set_title(f'{name} vs alpha', fontsize=12)

            try:
                sort_idx = np.argsort(x_vals)
                window = max(len(x_vals) // 10, 3)
                smoothed = uniform_filter1d(alpha_mean[sort_idx].astype(float), size=window)
                axes[i].plot(x_vals[sort_idx], smoothed, 'r-', lw=2, alpha=0.7, label='Trend')
                axes[i].legend()
            except:
                pass

        for i in range(n_celeb_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}celeb_features.png', dpi=150)
        plt.close()

    # === 图7: 舞者特征 vs delta（检测非线性） ===
    X_pro = td_train['X_pro']
    X_pro_names = td_train['X_pro_names']
    n_pro_features = len(X_pro_names)

    if n_pro_features > 0:
        n_cols_plot = min(3, n_pro_features)
        n_rows_plot = (n_pro_features + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(6*n_cols_plot, 5*n_rows_plot))
        axes = np.atleast_2d(axes).flatten()

        for i, name in enumerate(X_pro_names):
            x_vals = X_pro[:, i]
            axes[i].scatter(x_vals, delta_mean, alpha=0.5, s=20)
            axes[i].set_xlabel(name, fontsize=11)
            axes[i].set_ylabel('delta', fontsize=11)
            axes[i].set_title(f'{name} vs delta', fontsize=12)

            try:
                sort_idx = np.argsort(x_vals)
                window = max(len(x_vals) // 10, 3)
                smoothed = uniform_filter1d(delta_mean[sort_idx].astype(float), size=window)
                axes[i].plot(x_vals[sort_idx], smoothed, 'r-', lw=2, alpha=0.7, label='Trend')
                axes[i].legend()
            except:
                pass

        for i in range(n_pro_features, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}pro_features.png', dpi=150)
        plt.close()

    # === 图8: 名人级样条效果 ===
    celeb_spline_bases = train_datas.get('celeb_spline_bases')
    celeb_spline_feature_names = train_datas.get('celeb_spline_feature_names') or []
    if celeb_spline_bases and len(celeb_spline_bases) > 0:
        n_celeb_spline = len(celeb_spline_bases)
        fig, axes = plt.subplots(1, n_celeb_spline, figsize=(6*n_celeb_spline, 5))
        if n_celeb_spline == 1:
            axes = [axes]

        X_celeb_full = td_train['X_celeb']
        X_celeb_names_full = td_train['X_celeb_names']

        for i, feat_name in enumerate(celeb_spline_feature_names):
            key = f'celeb_spline_{i}_coef'
            if key in posterior and feat_name in X_celeb_names_full:
                col_idx = X_celeb_names_full.index(feat_name)
                x_vals = X_celeb_full[:, col_idx]
                coef = posterior[key].mean(axis=0)
                basis = celeb_spline_bases[i]
                y_spline = basis @ coef

                sort_idx = np.argsort(x_vals)
                axes[i].scatter(x_vals, y_spline, alpha=0.5, s=20, label='Data')
                axes[i].plot(x_vals[sort_idx], y_spline[sort_idx], 'r-', lw=2, label='Spline fit')
                axes[i].set_xlabel(feat_name)
                axes[i].set_ylabel('Spline Contribution to Alpha')
                axes[i].set_title(f'{feat_name} Nonlinear Effect (Celebrity)')
                axes[i].legend()
                axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/{prefix}celeb_spline_effects.png', dpi=150)
        plt.close()

    print(f"  [Diagnostics] Saved to {output_dir}/{prefix}*.png")

    # 打印贡献度摘要
    total_var = np.var(mu)
    print(f"  [Contribution Analysis]")
    print(f"    Alpha (Celebrity): var={np.var(alpha_contrib):.4f} ({100*np.var(alpha_contrib)/total_var:.1f}%)")
    print(f"    Delta (Pro):       var={np.var(delta_contrib):.4f} ({100*np.var(delta_contrib)/total_var:.1f}%)")
    print(f"    Linear Features:   var={np.var(linear_contrib):.4f} ({100*np.var(linear_contrib)/total_var:.1f}%)")
    print(f"    Spline (Nonlin):   var={np.var(spline_contrib):.4f} ({100*np.var(spline_contrib)/total_var:.1f}%)")
