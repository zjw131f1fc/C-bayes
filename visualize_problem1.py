"""
第一问主脚本
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem1_src.visualize import (
    plot_posterior_trajectory,
    plot_score_trajectory_with_threshold,
    plot_decision_gap,
    generate_consistency_table,
    find_controversial_and_certain_celebs,
    get_celebs_by_variance,
    get_eliminated_celebs,
    print_summary_stats,
    plot_posterior_std_bar,
)

OUTPUT_DIR = 'outputs/problem1'


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - 第一问可视化")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    datas = load_data('outputs/results/results.pkl')

    # 检查必要数据
    if 'P_fan_samples' not in datas:
        print("  Error: P_fan_samples not found. Please re-run main.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 打印汇总统计
    print("\n[2] 汇总统计...")
    print_summary_stats(datas)

    # 图表三：一致性总结表
    print("\n[3] 生成一致性总结表...")
    generate_consistency_table(datas, OUTPUT_DIR)

    # 图表二：决策缺口散点图
    print("\n[4] 生成决策缺口散点图...")
    plot_decision_gap(datas, OUTPUT_DIR)

    # 图表四：后验标准差柱状图
    print("\n[5] 生成后验标准差柱状图...")
    feature_names = datas.get('X_obs_names_filtered')
    plot_posterior_std_bar(datas, OUTPUT_DIR, feature_names)

    # 图表一：后验估计轨迹图
    print("\n[6] 生成后验估计轨迹图...")

    # 获取不同方差级别的选手（高、中、低各3个）
    celebs_to_plot = get_celebs_by_variance(datas, n_per_group=3)

    print(f"  将绘制 {len(celebs_to_plot)} 位选手的后验轨迹图:")
    for cid, var, group in celebs_to_plot:
        group_cn = {'high': '高不确定性', 'medium': '中等', 'low': '低不确定性'}[group]
        print(f"    - celeb_id={cid}, var={var:.6f} ({group_cn})")
        plot_posterior_trajectory(datas, f"{group}_celeb_{cid}", cid, OUTPUT_DIR)

    # 图表：得分轨迹图（带淘汰临界线）
    print("\n[7] 生成得分轨迹图（带淘汰临界线）...")
    print(f"  将绘制 {len(celebs_to_plot)} 位选手的得分轨迹图:")
    for cid, var, group in celebs_to_plot:
        group_cn = {'high': '高不确定性', 'medium': '中等', 'low': '低不确定性'}[group]
        print(f"    - celeb_id={cid}, var={var:.6f} ({group_cn})")
        plot_score_trajectory_with_threshold(datas, f"{group}_celeb_{cid}", cid, OUTPUT_DIR)

    # 图表：被淘汰选手的得分轨迹图
    print("\n[8] 生成被淘汰选手的得分轨迹图...")
    eliminated_celebs = get_eliminated_celebs(datas, n_celebs=15)
    print(f"  将绘制 {len(eliminated_celebs)} 位被淘汰选手的得分轨迹图:")
    for cid, elim_week, n_weeks in eliminated_celebs:
        print(f"    - celeb_id={cid}, 淘汰周={elim_week}, 参与周数={n_weeks}")
        plot_score_trajectory_with_threshold(datas, f"elim_celeb_{cid}_week{elim_week}", cid, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"所有输出保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
