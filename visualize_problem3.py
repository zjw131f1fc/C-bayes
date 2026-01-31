"""
第三问主脚本
分析职业舞者与名人特征对评委评分和粉丝投票的影响
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_data
from problem3_src.visualize import analyze_feature_significance

OUTPUT_DIR = 'outputs/problem3'


def main():
    print("=" * 60)
    print("MCM 2026 Problem C - 第三问可视化")
    print("特征对评委评分 vs 粉丝投票的影响分析")
    print("=" * 60)

    # 加载数据
    print("\n[1] 加载数据...")
    datas = load_data('outputs/results/results.pkl')

    # 检查必要数据
    if 'posterior_samples' not in datas:
        print("  Error: posterior_samples not found. Please run main.py first.")
        return

    if 'beta_obs' not in datas['posterior_samples']:
        print("  Error: beta_obs not found in posterior_samples.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 分析特征显著性
    print("\n[2] 分析特征显著性...")
    results = analyze_feature_significance(datas, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print(f"所有输出保存到: {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
