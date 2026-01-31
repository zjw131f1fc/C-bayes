"""
FPI 分析主脚本
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem2_src.fpi import (
    compute_fpi_both_methods,
    compute_variance_decomposition,
    plot_fpi_comparison,
    plot_fpi_by_season,
    plot_fpi_scatter,
    plot_power_shift,
    print_fpi_summary,
    generate_fpi_summary_table,
)

OUTPUT_DIR = 'outputs/problem2'


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - FPI 分析")
    print("粉丝权力指数 (Fan Power Index)")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    datas = load_data('outputs/results/results.pkl')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 计算 FPI
    print("\n[2] 计算 FPI...")
    fpi_data = compute_fpi_both_methods(datas)
    print(f"  分析了 {len(fpi_data)} 周的数据")

    # 打印汇总统计
    print_fpi_summary(fpi_data)

    # 生成图表
    print("\n[3] 生成 FPI 对比图...")
    plot_fpi_comparison(fpi_data, OUTPUT_DIR)

    print("\n[4] 生成 FPI 按赛季图...")
    plot_fpi_by_season(fpi_data, OUTPUT_DIR)

    print("\n[5] 生成 FPI 散点图...")
    plot_fpi_scatter(fpi_data, OUTPUT_DIR)

    # 计算方差分解
    print("\n[6] 计算方差分解...")
    var_data = compute_variance_decomposition(datas)

    print("\n[7] 生成影响力演变图 (Power Shift)...")
    plot_power_shift(var_data, OUTPUT_DIR)

    # 生成汇总表格
    print("\n[8] 生成汇总表格...")
    generate_fpi_summary_table(fpi_data, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"所有输出保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
