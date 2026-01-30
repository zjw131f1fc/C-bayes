"""
MCM 2026 Problem C - 与星共舞粉丝投票预测
分层贝叶斯模型 + GAM 框架
"""

# 强制使用 CPU（必须在 import jax 之前设置）
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
jax.config.update("jax_compilation_cache_dir", ".jax_cache")  # 启用编译缓存

import numpyro
numpyro.set_host_device_count(4)  # CPU 上模拟多设备以支持多链并行

import numpy as np
from scipy.special import softmax
from patsy import dmatrix
from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils import load_config, save_data
from src.preprocess import load_mock_data, load_data, validate_data, filter_data
from src.model import build_model, train, extract_posterior, compute_metrics, generate_output


def _build_spline_basis(x, n_knots, degree):
    """构建B样条基矩阵"""
    knots = np.linspace(x.min(), x.max(), n_knots)
    basis = dmatrix(
        f"bs(x, knots={list(knots[1:-1])}, degree={degree}, include_intercept=False) - 1",
        {"x": x}, return_type='dataframe'
    )
    return np.asarray(basis, dtype=np.float32)


def visualize_diagnostics(config, train_datas, test_datas, fold_idx, output_dir='outputs/diagnostics'):
    """诊断可视化：分析线性/非线性关系"""
    import os
    os.makedirs(output_dir, exist_ok=True)

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
        from src.model import _parse_interaction
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
    plt.savefig(f'{output_dir}/fold{fold_idx}_contributions.png', dpi=150)
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

            # 计算该特征的样条贡献
            key = f'spline_{i}_coef'
            if key in posterior:
                coef = posterior[key].mean(axis=0)
                basis = _build_spline_basis(x_vals, model_cfg['n_spline_knots'], model_cfg['spline_degree'])
                y_spline = basis @ coef

                # 散点图 + 样条曲线
                sort_idx = np.argsort(x_vals)
                axes[i].scatter(x_vals, y_spline, alpha=0.3, s=10, label='Data')
                axes[i].plot(x_vals[sort_idx], y_spline[sort_idx], 'r-', lw=2, label='Spline fit')
                axes[i].set_xlabel(feature_name)
                axes[i].set_ylabel('Spline Contribution')
                axes[i].set_title(f'{feature_name} Nonlinear Effect')
                axes[i].legend()
                axes[i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold{fold_idx}_spline_effects.png', dpi=150)
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

        # 添加LOWESS平滑线来检测非线性
        try:
            from scipy.ndimage import uniform_filter1d
            sort_idx = np.argsort(x_vals)
            window = max(len(x_vals) // 20, 5)
            smoothed = uniform_filter1d(mu[sort_idx].astype(float), size=window)
            axes[i].plot(x_vals[sort_idx], smoothed, 'r-', lw=2, alpha=0.7, label='Trend')
            axes[i].legend()
        except:
            pass

    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fold{fold_idx}_feature_vs_mu.png', dpi=150)
    plt.close()

    # === 图4: 测试集预测残差分析 ===
    S_samples = test_datas['S_samples']
    S_mean = S_samples.mean(axis=0)

    # 计算每周的排名残差
    week_data = td_test['week_data']
    residuals = []
    actual_ranks = []
    pred_ranks = []

    for wd in week_data:
        if wd['n_eliminated'] == 0:
            continue
        mask = wd['obs_mask']
        S_week = S_mean[mask]

        # 预测排名
        pred_rank = np.argsort(np.argsort(S_week))

        # 实际淘汰者
        elim_mask = wd['eliminated_mask'][mask]
        actual_rank = np.zeros_like(pred_rank)
        actual_rank[elim_mask] = 1  # 淘汰者标记为1

        residuals.extend(pred_rank[elim_mask] - 0)  # 淘汰者的预测排名应该是0
        pred_ranks.extend(pred_rank)
        actual_ranks.extend(actual_rank)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 预测排名分布（淘汰者 vs 非淘汰者）
    pred_ranks = np.array(pred_ranks)
    actual_ranks = np.array(actual_ranks)

    axes[0].hist(pred_ranks[actual_ranks == 1], bins=20, alpha=0.7, label='Eliminated', density=True)
    axes[0].hist(pred_ranks[actual_ranks == 0], bins=20, alpha=0.7, label='Survived', density=True)
    axes[0].set_xlabel('Predicted Rank (0=lowest)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Eliminated vs Survived Rank Distribution')
    axes[0].legend()

    # 残差直方图
    axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Eliminated Pred Rank (ideal=0)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Elimination Residuals\nmean={np.mean(residuals):.2f}, median={np.median(residuals):.0f}')
    axes[1].axvline(x=0, color='red', linestyle='--', label='Ideal')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fold{fold_idx}_residuals.png', dpi=150)
    plt.close()

    # === 图5: 线性特征的效应 ===
    if linear_cols:
        n_lin = len(linear_cols)
        n_cols_plot = min(3, n_lin)
        n_rows_plot = (n_lin + n_cols_plot - 1) // n_cols_plot
        fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(5*n_cols_plot, 4*n_rows_plot))
        axes = np.atleast_2d(axes).flatten()

        if 'beta_obs' in posterior:
            beta_samples = posterior['beta_obs']  # [n_samples, n_features]
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
        plt.savefig(f'{output_dir}/fold{fold_idx}_linear_effects.png', dpi=150)
        plt.close()

    print(f"  [Diagnostics] Saved to {output_dir}/fold{fold_idx}_*.png")

    # 打印贡献度摘要
    total_var = np.var(mu)
    print(f"  [Contribution Analysis]")
    print(f"    Alpha (Celebrity): var={np.var(alpha_contrib):.4f} ({100*np.var(alpha_contrib)/total_var:.1f}%)")
    print(f"    Delta (Pro):       var={np.var(delta_contrib):.4f} ({100*np.var(delta_contrib)/total_var:.1f}%)")
    print(f"    Linear Features:   var={np.var(linear_contrib):.4f} ({100*np.var(linear_contrib)/total_var:.1f}%)")
    print(f"    Spline (Nonlin):   var={np.var(spline_contrib):.4f} ({100*np.var(spline_contrib)/total_var:.1f}%)")


def predict(config, train_datas, eval_datas):
    """用后验样本对 eval_datas 计算 S"""
    posterior = train_datas['posterior_samples']
    td = eval_datas['train_data']
    model_cfg = config['model']

    n_samples = posterior['alpha'].shape[0]
    n_obs = td['n_obs']

    # 索引
    celeb_idx = td['celeb_idx']
    pro_idx = td['pro_idx']

    # 获取观测特征配置
    X_obs_orig = td['X_obs'].copy()
    X_obs_names_orig = list(td['X_obs_names'])
    X_obs = X_obs_orig.copy()
    X_obs_names = list(X_obs_names_orig)
    spline_features = model_cfg['spline_features'] or []
    exclude_obs = model_cfg.get('exclude_obs_features') or []
    obs_interaction = model_cfg.get('obs_interaction_features') or []
    center_features = model_cfg.get('center_features', False)

    # 添加观测特征交互项
    if obs_interaction:
        from src.model import _parse_interaction
        for expr in obs_interaction:
            new_col, new_name = _parse_interaction(expr, X_obs_orig, X_obs_names_orig)
            X_obs = np.column_stack([X_obs, new_col])
            X_obs_names.append(new_name)

    # 过滤掉排除的观测特征
    if exclude_obs:
        keep_cols = [i for i, name in enumerate(X_obs_names) if name not in exclude_obs]
        X_obs = X_obs[:, keep_cols]
        X_obs_names = [X_obs_names[i] for i in keep_cols]

    # 使用训练集的均值进行中心化
    feature_means = train_datas.get('feature_means')
    if center_features and feature_means is not None:
        X_obs = X_obs - feature_means

    # 分离线性/样条特征
    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    X_lin = X_obs[:, linear_cols] if linear_cols else None
    spline_bases = [_build_spline_basis(X_obs[:, c], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree']) for c in spline_cols]

    # 对每个后验样本计算 S
    S_samples = np.zeros((n_samples, n_obs), dtype=np.float32)

    for s in tqdm(range(n_samples), desc="Predicting"):
        # mu = alpha[celeb_idx] + delta[pro_idx] + linear + spline
        mu = posterior['alpha'][s, celeb_idx] + posterior['delta'][s, pro_idx]

        if X_lin is not None and 'beta_obs' in posterior:
            mu = mu + X_lin @ posterior['beta_obs'][s]

        for i, basis in enumerate(spline_bases):
            key = f'spline_{i}_coef'
            if key in posterior:
                mu = mu + basis @ posterior[key][s]

        # P_fan = softmax(mu) per week
        P_fan = np.zeros(n_obs, dtype=np.float32)
        for wd in td['week_data']:
            mask = wd['obs_mask']
            P_fan[mask] = softmax(mu[mask])

        # S = judge_score + fan_score
        S = np.zeros(n_obs, dtype=np.float32)
        for wd in td['week_data']:
            mask = wd['obs_mask']
            if wd['rule_method'] == 1:  # 百分比法
                S[mask] = td['judge_score_pct'][mask] + P_fan[mask]
            else:  # 排名法
                P_week = P_fan[mask]
                n_contestants = len(P_week)
                if n_contestants > 1:
                    diff = P_week[:, None] - P_week[None, :]
                    soft_rank = np.sum(1 / (1 + np.exp(-diff / 0.1)), axis=1) - 0.5
                    R_fan = soft_rank / (n_contestants - 1)
                else:
                    R_fan = np.array([0.5], dtype=np.float32)
                S[mask] = td['judge_rank_score'][mask] + R_fan

        S_samples[s] = S

    eval_datas['S_samples'] = S_samples
    return eval_datas


def run_single(config, datas):
    """单次训练（使用全部数据）"""
    model = build_model(config, datas)
    datas = train(config, model, datas)
    datas = extract_posterior(config, datas)
    datas = predict(config, datas, datas)  # eval_datas = train_datas
    datas = compute_metrics(config, datas)
    datas = generate_output(config, datas)
    return model, datas


def run_cv(config, datas):
    """留一赛季交叉验证"""
    n_seasons = datas['train_data']['n_seasons']
    cv_results = []

    for test_season in range(n_seasons):
        test_season = 10
        print(f"\n=== CV Fold {test_season + 1}/{n_seasons}: 测试赛季 {test_season} ===")

        print("  [DEBUG] filter_data...")
        train_datas, test_datas = filter_data(datas, test_season)
        print(f"  [DEBUG] train: {train_datas['train_data']['n_obs']} obs, test: {test_datas['train_data']['n_obs']} obs")

        # 训练
        print("  [DEBUG] build_model...")
        model = build_model(config, train_datas)
        print("  [DEBUG] train...")
        train_datas = train(config, model, train_datas)
        print("  [DEBUG] extract_posterior...")
        train_datas = extract_posterior(config, train_datas)

        # 预测 + 评估
        print("  [DEBUG] predict...")
        test_datas = predict(config, train_datas, test_datas)
        print("  [DEBUG] compute_metrics...")
        test_datas = compute_metrics(config, test_datas)

        cv_results.append({
            'test_season': test_season,
            'metrics': test_datas.get('metrics', {}),
        })

        # 第一折结束后画诊断图并保存结果
        if test_season == 10:
            print("  [DEBUG] visualize_diagnostics...")
            visualize_diagnostics(config, train_datas, test_datas, fold_idx=10)
            # 保存结果用于分析
            fold0_results = {
                'train_datas': train_datas,
                'test_datas': test_datas,
                'config': config,
            }
            save_data(fold0_results, 'fold0_results.pkl')
            print("  [DEBUG] Saved fold0_results.pkl")

    return cv_results


if __name__ == "__main__":
    config = load_config('config.yaml')
    datas = {}

    # 加载数据
    datas = load_data('datas.pkl', datas)
    # datas = load_mock_data(config, datas)   # 测试用模拟数据

    # 校验数据
    datas = validate_data(datas)

    # # 1. 先全量训练（快速得到结果）
    # print("\n=== 全量训练 ===")
    # model, datas = run_single(config, datas)

    # # 2. 保存初步结果（即使CV中断也有结果）
    # save_data(datas, 'results.pkl')
    # print("初步结果已保存到 results.pkl")

    # 3. 最后再跑交叉验证（耗时）
    print("\n=== 交叉验证 ===")
    cv_results = run_cv(config, datas)
    print(f"\nCV Results: {cv_results}")

    # 4. 保存包含CV结果的完整数据
    datas['cv_results'] = cv_results
    save_data(datas, 'results_with_cv.pkl')
    print("Done!")
