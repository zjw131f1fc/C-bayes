"""诊断可视化模块"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from patsy import dmatrix


def compute_rhat(samples):
    """计算 Gelman-Rubin R-hat 统计量

    Args:
        samples: shape [n_chains, n_samples] 或 [n_chains, n_samples, n_params]

    Returns:
        R-hat 值（标量或数组）
    """
    if samples.ndim == 2:
        # [n_chains, n_samples] -> 单个参数
        n_chains, n_samples = samples.shape
        chain_means = samples.mean(axis=1)  # [n_chains]
        chain_vars = samples.var(axis=1, ddof=1)  # [n_chains]

        # 链间方差 B
        B = n_samples * np.var(chain_means, ddof=1)
        # 链内方差 W
        W = np.mean(chain_vars)

        # 后验方差估计
        var_hat = (n_samples - 1) / n_samples * W + B / n_samples
        # R-hat
        rhat = np.sqrt(var_hat / W) if W > 1e-10 else 1.0
        return rhat

    elif samples.ndim == 3:
        # [n_chains, n_samples, n_params] -> 多个参数
        n_chains, n_samples, n_params = samples.shape
        rhats = np.zeros(n_params)
        for i in range(n_params):
            rhats[i] = compute_rhat(samples[:, :, i])
        return rhats

    else:
        raise ValueError(f"samples must be 2D or 3D, got {samples.ndim}D")


def compute_mcmc_diagnostics(trace, n_chains):
    """计算 MCMC 诊断统计量

    Args:
        trace: dict, 后验样本 {param_name: samples}，samples shape [n_total_samples, ...]
        n_chains: int, 链数

    Returns:
        dict: 诊断结果
    """
    diagnostics = {}

    for param_name, samples in trace.items():
        if param_name in ['P_fan', 'S']:  # 跳过确定性变量
            continue

        n_total = samples.shape[0]
        n_samples_per_chain = n_total // n_chains

        if n_samples_per_chain < 10:
            continue

        # 重塑为 [n_chains, n_samples_per_chain, ...]
        if samples.ndim == 1:
            # 标量参数
            samples_reshaped = samples[:n_chains * n_samples_per_chain].reshape(n_chains, n_samples_per_chain)
            rhat = compute_rhat(samples_reshaped)
            diagnostics[param_name] = {
                'rhat': rhat,
                'mean': samples.mean(),
                'std': samples.std(),
            }
        elif samples.ndim == 2:
            # 向量参数 [n_total, n_params]
            n_params = samples.shape[1]
            samples_reshaped = samples[:n_chains * n_samples_per_chain].reshape(n_chains, n_samples_per_chain, n_params)
            rhats = compute_rhat(samples_reshaped)
            diagnostics[param_name] = {
                'rhat': rhats,
                'rhat_max': rhats.max(),
                'rhat_mean': rhats.mean(),
                'mean': samples.mean(axis=0),
                'std': samples.std(axis=0),
            }

    return diagnostics


def print_rhat_summary(diagnostics):
    """打印 R-hat 诊断摘要"""
    print("\n" + "=" * 60)
    print("Gelman-Rubin Diagnostic (R-hat)")
    print("=" * 60)
    print("R-hat < 1.01: Excellent convergence")
    print("R-hat < 1.05: Good convergence")
    print("R-hat < 1.10: Acceptable convergence")
    print("R-hat > 1.10: Poor convergence (consider more samples)")
    print("-" * 60)

    # 收集所有 R-hat 值
    all_rhats = []
    for param_name, diag in diagnostics.items():
        rhat = diag['rhat']
        if np.isscalar(rhat):
            all_rhats.append(rhat)
            status = "OK" if rhat < 1.05 else ("WARN" if rhat < 1.10 else "BAD")
            print(f"{param_name:30s}: R-hat = {rhat:.4f} [{status}]")
        else:
            all_rhats.extend(rhat.flatten())
            rhat_max = diag['rhat_max']
            rhat_mean = diag['rhat_mean']
            status = "OK" if rhat_max < 1.05 else ("WARN" if rhat_max < 1.10 else "BAD")
            print(f"{param_name:30s}: R-hat max = {rhat_max:.4f}, mean = {rhat_mean:.4f} [{status}]")

    print("-" * 60)
    all_rhats = np.array(all_rhats)
    print(f"{'Overall':30s}: max = {all_rhats.max():.4f}, mean = {all_rhats.mean():.4f}")
    print(f"{'Parameters with R-hat > 1.05':30s}: {(all_rhats > 1.05).sum()} / {len(all_rhats)}")
    print(f"{'Parameters with R-hat > 1.10':30s}: {(all_rhats > 1.10).sum()} / {len(all_rhats)}")
    print("=" * 60)

    return all_rhats


def plot_rhat_histogram(diagnostics, output_dir):
    """绘制 R-hat 直方图"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    all_rhats = []
    for param_name, diag in diagnostics.items():
        rhat = diag['rhat']
        if np.isscalar(rhat):
            all_rhats.append(rhat)
        else:
            all_rhats.extend(rhat.flatten())

    all_rhats = np.array(all_rhats)

    plt.figure(figsize=(10, 6))
    plt.hist(all_rhats, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=1.0, color='green', linestyle='-', linewidth=2, label='R-hat = 1.0 (ideal)')
    plt.axvline(x=1.05, color='orange', linestyle='--', linewidth=2, label='R-hat = 1.05 (good)')
    plt.axvline(x=1.10, color='red', linestyle='--', linewidth=2, label='R-hat = 1.10 (threshold)')
    plt.xlabel('R-hat', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Gelman-Rubin R-hat Distribution\n(n={len(all_rhats)}, max={all_rhats.max():.4f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rhat_histogram.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/rhat_histogram.png")


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

    # 直接使用 train_data 中已处理好的特征（build_model 已处理交互项、排除、中心化）
    X_lin = td_train.get('X_lin')
    linear_names = td_train.get('linear_feature_names', [])
    spline_bases = td_train.get('spline_bases', [])
    spline_names = td_train.get('spline_feature_names', [])

    # === 图1: 各部分贡献的分布 ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 计算训练集上的各部分贡献
    alpha_mean = posterior['alpha'].mean(axis=0)
    delta_mean = posterior['delta'].mean(axis=0)

    alpha_contrib = alpha_mean[td_train['celeb_idx']]
    delta_contrib = delta_mean[td_train['pro_idx']]

    linear_contrib = np.zeros(td_train['n_obs'])
    if X_lin is not None and 'beta_obs' in posterior:
        beta_obs = posterior['beta_obs'].mean(axis=0)
        linear_contrib = X_lin @ beta_obs

    spline_contrib = np.zeros(td_train['n_obs'])
    for i, basis in enumerate(spline_bases):
        key = f'spline_{i}_coef'
        if key in posterior:
            coef = posterior[key].mean(axis=0)
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
    # 获取原始特征值用于绘图（从 X_obs 中提取样条特征列）
    X_obs = td_train['X_obs']
    X_obs_names = td_train['X_obs_names']

    n_spline = len(spline_bases)
    if n_spline > 0:
        fig, axes = plt.subplots(1, n_spline, figsize=(6*n_spline, 5))
        if n_spline == 1:
            axes = [axes]

        for i, (basis, feature_name) in enumerate(zip(spline_bases, spline_names)):
            # 从 X_obs 中找到对应的原始特征列
            if feature_name in X_obs_names:
                col_idx = X_obs_names.index(feature_name)
                x_vals = X_obs[:, col_idx]
            else:
                x_vals = np.arange(len(basis))  # fallback

            key = f'spline_{i}_coef'
            if key in posterior:
                coef = posterior[key].mean(axis=0)
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

    # 使用线性特征绘图
    n_features = len(linear_names)
    if n_features > 0 and X_lin is not None:
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        axes = np.atleast_2d(axes).flatten()

        for i, name in enumerate(linear_names):
            x_vals = X_lin[:, i]
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
    if linear_names and len(linear_names) > 0:
        n_lin = len(linear_names)
        n_cols_plot = min(3, n_lin)
        n_rows_plot = (n_lin + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5*n_cols_plot, 4*n_rows_plot))
        axes = np.atleast_2d(axes).flatten()

        if 'beta_obs' in posterior:
            beta_samples = posterior['beta_obs']
            for i, name in enumerate(linear_names):
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
