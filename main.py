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

from src.utils import load_config, save_data
from src.preprocess import load_mock_data, load_data, validate_data, filter_data, normalize_judge_score_weekly
from src.model import build_model, train, extract_posterior, compute_metrics, generate_output, predict
from src.visualize import visualize_diagnostics


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
        # test_season = 19  # 固定测试某个赛季
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

        # 特定折画诊断图并保存结果
        if test_season == 10:
            print("  [DEBUG] visualize_diagnostics...")
            visualize_diagnostics(config, train_datas, test_datas, fold_idx=10)
            # 保存结果用于分析
            fold0_results = {
                'train_datas': train_datas,
                'test_datas': test_datas,
                'config': config,
            }
            save_data(fold0_results, 'outputs/results/fold0_results.pkl')
            print("  [DEBUG] Saved outputs/results/fold0_results.pkl")

    return cv_results


if __name__ == "__main__":
    config = load_config('config.yaml')
    datas = {}

    # 加载数据
    datas = load_data('data/datas.pkl', datas)
    # datas = load_mock_data(config, datas)   # 测试用模拟数据

    # 校验数据
    datas = validate_data(datas)

    # 周内标准化评委分（替换 judge_score_season_zscore）
    datas = normalize_judge_score_weekly(datas)

    # 1. 全量训练（在整个数据集上训练并验证）
    print("\n=== 全量训练 ===")
    model, datas = run_single(config, datas)

    # 2. 诊断可视化
    visualize_diagnostics(config, datas, datas)

    # 3. 保存结果
    save_data(datas, 'outputs/results/results.pkl')
    print("结果已保存到 outputs/results/results.pkl")

    # # 4. 交叉验证（可选）
    # print("\n=== 交叉验证 ===")
    # cv_results = run_cv(config, datas)
    # print(f"\nCV Results: {cv_results}")

    # # 5. 保存包含CV结果的完整数据
    # datas['cv_results'] = cv_results
    # save_data(datas, 'outputs/results/results_with_cv.pkl')
    print("Done!")
