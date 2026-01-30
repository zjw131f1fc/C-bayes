"""
MCM 2026 Problem C - 与星共舞粉丝投票预测
分层贝叶斯模型 + GAM 框架
"""

# 强制使用 CPU（必须在 import jax 之前设置）
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import numpy as np
from scipy.special import softmax
from patsy import dmatrix
from tqdm import tqdm

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

    # 分离线性/样条特征
    X_obs = td['X_obs']
    X_obs_names = td['X_obs_names']
    spline_features = model_cfg['spline_features']

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

    return cv_results


if __name__ == "__main__":
    config = load_config('config.yaml')
    datas = {}

    # 加载数据
    datas = load_data('datas.pkl', datas)
    # datas = load_mock_data(config, datas)   # 测试用模拟数据

    # 校验数据
    datas = validate_data(datas)

    # 1. 先全量训练（快速得到结果）
    print("\n=== 全量训练 ===")
    model, datas = run_single(config, datas)

    # 2. 保存初步结果（即使CV中断也有结果）
    save_data(datas, 'results.pkl')
    print("初步结果已保存到 results.pkl")

    # 3. 最后再跑交叉验证（耗时）
    print("\n=== 交叉验证 ===")
    cv_results = run_cv(config, datas)
    print(f"\nCV Results: {cv_results}")

    # 4. 保存包含CV结果的完整数据
    datas['cv_results'] = cv_results
    save_data(datas, 'results_with_cv.pkl')
    print("Done!")
