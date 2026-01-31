"""
第二问主脚本
比较排名法和百分比法的差异
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem2_src.visualize import (
    compute_scores_both_methods,
    compute_rule_comparison_metrics,
    print_summary_stats,
    generate_summary_table,
    plot_rule_heatmap,
    plot_bump_chart,
    find_interesting_celebs,
)

OUTPUT_DIR = 'outputs/problem2'


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - 第二问可视化")
    print("规则比较：排名法 vs 百分比法")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    datas = load_data('outputs/results/results.pkl')

    # 检查必要数据
    if 'pfan_filtered' not in datas and 'P_fan_samples' not in datas:
        print("  Error: Neither pfan_filtered nor P_fan_samples found.")
        print("  Please run main.py and filter_posterior.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 计算两种方法的得分
    print("\n[2] 计算两种规则的得分...")
    S_pct_mean, S_rank_mean = compute_scores_both_methods(datas)
    print(f"  S_pct_mean shape: {S_pct_mean.shape}")
    print(f"  S_rank_mean shape: {S_rank_mean.shape}")

    # 计算比较指标
    print("\n[3] 计算比较指标...")
    metrics, week_details = compute_rule_comparison_metrics(datas, S_pct_mean, S_rank_mean)

    # 打印汇总统计
    print_summary_stats(datas, metrics)

    # 生成汇总表格
    print("\n[4] 生成汇总表格...")
    generate_summary_table(metrics, week_details, OUTPUT_DIR)

    # 规则偏差热力图
    print("\n[5] 生成规则偏差热力图...")
    plot_rule_heatmap(week_details, OUTPUT_DIR)

    # 找到排名差异最大的选手
    print("\n[6] 生成淘汰轨迹对比图 (Bump Chart)...")
    interesting_celebs = find_interesting_celebs(datas, week_details, n_celebs=5)
    print(f"  排名差异最大的选手:")
    for cid, avg_diff, n_weeks in interesting_celebs:
        print(f"    - celeb_id={cid}, avg_diff={avg_diff:.2f}, n_weeks={n_weeks}")
        plot_bump_chart(datas, week_details, cid, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"所有输出保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
