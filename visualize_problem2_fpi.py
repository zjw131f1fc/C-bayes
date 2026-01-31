"""
第二问 FPI 分析主脚本
粉丝权力指数 (Fan Power Index) 计算和可视化

FPI = Var(S_fan) / Var(S_total)
- 如果 FPI > 0.5，意味着该周的淘汰结果主要由粉丝偏好驱动
- 如果 FPI 接近 1，评委分则沦为陪衬
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem2_src.fpi import (
    compute_fpi_both_methods,
    plot_fpi_comparison,
    plot_fpi_by_season,
    plot_fpi_scatter,
    plot_fpi_difference_by_week,
    find_high_fpi_weeks,
    generate_fpi_summary_table,
    print_fpi_summary,
)

OUTPUT_DIR = 'outputs/problem2'


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - Fan Power Index (FPI) Analysis")
    print("=" * 60)

    # 加载数据
    print("\n[1] Loading data...")
    datas = load_data('outputs/results/results.pkl')

    # 检查必要数据
    if 'P_fan_samples' not in datas:
        print("  Error: P_fan_samples not found. Please re-run main.py first.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 计算 FPI
    print("\n[2] Computing Fan Power Index (FPI)...")
    print("    - Percentage Method: FPI = Var(P_fan) / Var(P_fan + judge_score_pct)")
    print("    - Ranking Method:    FPI = Var(R_fan) / Var(R_fan + judge_rank_score)")
    fpi_data = compute_fpi_both_methods(datas)
    print(f"    Computed FPI for {len(fpi_data)} weeks")

    # 打印汇总统计
    print_fpi_summary(fpi_data)

    # 生成汇总表格
    print("\n[3] Generating FPI summary table...")
    generate_fpi_summary_table(fpi_data, OUTPUT_DIR)

    # 绘制 FPI 对比图
    print("\n[4] Plotting FPI comparison (boxplot & histogram)...")
    plot_fpi_comparison(fpi_data, OUTPUT_DIR)

    # 按赛季展示 FPI
    print("\n[5] Plotting FPI by season...")
    plot_fpi_by_season(fpi_data, OUTPUT_DIR)

    # FPI 散点图
    print("\n[6] Plotting FPI scatter (Percentage vs Ranking)...")
    plot_fpi_scatter(fpi_data, OUTPUT_DIR)

    # FPI 差异随周次变化
    print("\n[7] Plotting FPI difference by week...")
    plot_fpi_difference_by_week(fpi_data, OUTPUT_DIR)

    # 找到高 FPI 周
    print("\n[8] Identifying high FPI weeks (controversial weeks)...")
    high_fpi_weeks = find_high_fpi_weeks(fpi_data, threshold=0.7, n_top=10)
    if high_fpi_weeks:
        print(f"    Found {len(high_fpi_weeks)} weeks with FPI (Percentage) > 0.7:")
        for hw in high_fpi_weeks:
            print(f"      Season {hw['season']}, Week {hw['week']}: "
                  f"FPI_pct={hw['fpi_pct']:.3f}, FPI_rank={hw['fpi_rank']:.3f}, "
                  f"diff={hw['diff']:.3f}")

    print("\n" + "=" * 60)
    print(f"All outputs saved to: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
