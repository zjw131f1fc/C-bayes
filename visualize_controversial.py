"""
争议选手可视化主脚本
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem2_src.controversial import (
    plot_bump_chart_single,
    plot_rank_diff_bar,
    plot_survival_curves,
    plot_scenario_heatmap,
    simulate_season_4scenarios,
    compute_rankings_both_methods,
)

OUTPUT_DIR = 'outputs/problem2'


# 争议选手列表
# (celeb_id, celeb_name, data_season, actual_season, description)
CONTROVERSIAL_CELEBS = [
    (176, 'Jerry Rice', 1, 2, '5周评委最低分，亚军'),
    (41, 'Billy Ray Cyrus', 3, 4, '6周评委最低分，第5名'),
    (48, 'Bristol Palin', 10, 11, '12次评委最低分，第3名'),
    (43, 'Bobby Bones', 26, 27, '评委评分偏低，冠军'),
]


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - 争议选手分析")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    datas = load_data('outputs/results/results.pkl')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 为每个争议选手生成图表
    print("\n[2] 生成排名轨迹对比图 (Bump Chart)...")
    for celeb_id, name, data_season, actual_season, desc in CONTROVERSIAL_CELEBS:
        print(f"\n  {name} (S{actual_season}): {desc}")
        plot_bump_chart_single(datas, celeb_id, name, data_season,
                               actual_season, OUTPUT_DIR)

    print("\n[3] 生成排名差异柱状图...")
    for celeb_id, name, data_season, actual_season, desc in CONTROVERSIAL_CELEBS:
        plot_rank_diff_bar(datas, celeb_id, name, data_season,
                           actual_season, OUTPUT_DIR)

    # 生存曲线（4种场景）
    print("\n[4] 生成生存曲线对比图 (4种场景)...")
    all_sim_results = {}
    for celeb_id, name, data_season, actual_season, desc in CONTROVERSIAL_CELEBS:
        print(f"\n  模拟 {name} (S{actual_season})...")
        sim_results = simulate_season_4scenarios(datas, data_season, n_simulations=1000)
        if sim_results and celeb_id in sim_results:
            all_sim_results[celeb_id] = sim_results[celeb_id]
            plot_survival_curves(datas, celeb_id, name, data_season,
                                 actual_season, OUTPUT_DIR, n_simulations=1000)

    # 场景热力图
    print("\n[5] 生成场景热力图...")
    celeb_info = [(cid, name, actual_season)
                  for cid, name, _, actual_season, _ in CONTROVERSIAL_CELEBS]
    plot_scenario_heatmap(all_sim_results, celeb_info, OUTPUT_DIR)

    # 打印汇总统计
    print("\n[6] 汇总统计...")
    print("\n" + "=" * 90)
    print("争议选手：4种场景下的期望淘汰周")
    print("=" * 90)
    print(f"{'选手':<20} {'Rank Only':<12} {'Pct Only':<12} {'Rank+Save':<12} {'Pct+Save':<12}")
    print("-" * 90)

    import numpy as np
    for celeb_id, name, data_season, actual_season, desc in CONTROVERSIAL_CELEBS:
        if celeb_id in all_sim_results:
            res = all_sim_results[celeb_id]
            a = np.mean(res['A']) if res['A'] else 0
            b = np.mean(res['B']) if res['B'] else 0
            c = np.mean(res['C']) if res['C'] else 0
            d = np.mean(res['D']) if res['D'] else 0
            print(f"{name:<20} {a:<12.1f} {b:<12.1f} {c:<12.1f} {d:<12.1f}")

    print("=" * 90)
    print("\n注: 数值越大表示存活越久")

    print("\n" + "=" * 60)
    print(f"所有输出保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
