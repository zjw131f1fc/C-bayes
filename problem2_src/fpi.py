"""
粉丝权力指数 (Fan Power Index, FPI) 计算和可视化

FPI = Var(S_fan) / Var(S_total)

- 百分比法: S_fan = P_fan, S_judge = judge_score_pct
- 排名法: S_fan = R_fan (粉丝排名分), S_judge = judge_rank_score
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150


def compute_fpi_both_methods(datas):
    """
    对每周，分别用排名法和百分比法计算 FPI

    FPI = Var(S_fan) / Var(S_total)

    百分比法: S_fan = P_fan, S_judge = judge_score_pct, S_total = S_fan + S_judge
    排名法: S_fan = R_fan, S_judge = judge_rank_score, S_total = S_fan + S_judge

    返回:
        fpi_data: list of dict, 每周的 FPI 数据
    """
    P_fan_samples = datas['P_fan_samples']  # [n_samples, n_obs]
    td = datas['train_data']
    week_data = td['week_data']
    n_samples, n_obs = P_fan_samples.shape

    fpi_data = []

    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        n_contestants = len(indices)
        season = wd['season']
        week = wd['week']

        if n_contestants < 2:
            continue

        # 获取该周的数据
        judge_score_pct = td['judge_score_pct'][mask]  # [n_contestants]
        judge_rank_score = td['judge_rank_score'][mask]  # [n_contestants]

        # 对每个后验样本计算 FPI，然后取均值
        fpi_pct_samples = []
        fpi_rank_samples = []

        for s in range(n_samples):
            P_fan = P_fan_samples[s, mask]  # [n_contestants]

            # === 百分比法 ===
            S_fan_pct = P_fan
            S_judge_pct = judge_score_pct
            S_total_pct = S_fan_pct + S_judge_pct

            var_fan_pct = np.var(S_fan_pct)
            var_total_pct = np.var(S_total_pct)

            if var_total_pct > 1e-10:
                fpi_pct = var_fan_pct / var_total_pct
            else:
                fpi_pct = np.nan
            fpi_pct_samples.append(fpi_pct)

            # === 排名法 ===
            # 计算粉丝排名分 R_fan (软排名)
            if n_contestants > 1:
                diff = P_fan[:, None] - P_fan[None, :]
                soft_rank = np.sum(1 / (1 + np.exp(-diff / 0.1)), axis=1) - 0.5
                R_fan = soft_rank / (n_contestants - 1)
            else:
                R_fan = np.array([0.5], dtype=np.float32)

            S_fan_rank = R_fan
            S_judge_rank = judge_rank_score
            S_total_rank = S_fan_rank + S_judge_rank

            var_fan_rank = np.var(S_fan_rank)
            var_total_rank = np.var(S_total_rank)

            if var_total_rank > 1e-10:
                fpi_rank = var_fan_rank / var_total_rank
            else:
                fpi_rank = np.nan
            fpi_rank_samples.append(fpi_rank)

        # 计算后验均值和标准差
        fpi_pct_samples = np.array(fpi_pct_samples)
        fpi_rank_samples = np.array(fpi_rank_samples)

        fpi_pct_mean = np.nanmean(fpi_pct_samples)
        fpi_pct_std = np.nanstd(fpi_pct_samples)
        fpi_rank_mean = np.nanmean(fpi_rank_samples)
        fpi_rank_std = np.nanstd(fpi_rank_samples)

        fpi_data.append({
            'season': season,
            'week': week,
            'n_contestants': n_contestants,
            'fpi_pct_mean': fpi_pct_mean,
            'fpi_pct_std': fpi_pct_std,
            'fpi_rank_mean': fpi_rank_mean,
            'fpi_rank_std': fpi_rank_std,
            'fpi_pct_samples': fpi_pct_samples,
            'fpi_rank_samples': fpi_rank_samples,
        })

    return fpi_data


def plot_fpi_comparison(fpi_data, output_dir):
    """
    对比两种方法的 FPI 分布（箱线图）
    """
    setup_plot_style()

    fpi_pct_all = [d['fpi_pct_mean'] for d in fpi_data if not np.isnan(d['fpi_pct_mean'])]
    fpi_rank_all = [d['fpi_rank_mean'] for d in fpi_data if not np.isnan(d['fpi_rank_mean'])]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：箱线图对比
    ax1 = axes[0]
    bp = ax1.boxplot([fpi_pct_all, fpi_rank_all],
                     labels=['Percentage Method', 'Ranking Method'],
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_alpha(0.7)

    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='FPI = 0.5')
    ax1.set_ylabel('Fan Power Index (FPI)', fontsize=12)
    ax1.set_title('FPI Distribution: Percentage vs Ranking Method', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # 右图：直方图对比
    ax2 = axes[1]
    bins = np.linspace(0, 1, 21)
    ax2.hist(fpi_pct_all, bins=bins, alpha=0.6, label='Percentage Method', color='steelblue')
    ax2.hist(fpi_rank_all, bins=bins, alpha=0.6, label='Ranking Method', color='coral')
    ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='FPI = 0.5')
    ax2.set_xlabel('Fan Power Index (FPI)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('FPI Histogram: Percentage vs Ranking Method', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_fpi_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_fpi_comparison.png")

    # 打印统计信息
    print(f"\n  FPI Statistics:")
    print(f"    Percentage Method: mean={np.mean(fpi_pct_all):.3f}, std={np.std(fpi_pct_all):.3f}")
    print(f"    Ranking Method:    mean={np.mean(fpi_rank_all):.3f}, std={np.std(fpi_rank_all):.3f}")
    print(f"    Percentage > 0.5:  {sum(1 for x in fpi_pct_all if x > 0.5)}/{len(fpi_pct_all)} ({100*sum(1 for x in fpi_pct_all if x > 0.5)/len(fpi_pct_all):.1f}%)")
    print(f"    Ranking > 0.5:     {sum(1 for x in fpi_rank_all if x > 0.5)}/{len(fpi_rank_all)} ({100*sum(1 for x in fpi_rank_all if x > 0.5)/len(fpi_rank_all):.1f}%)")


def plot_fpi_by_season(fpi_data, output_dir):
    """
    按赛季展示 FPI，特别标注 S27 (Bobby Bones) 赛季
    """
    setup_plot_style()

    # 按赛季分组
    season_fpi = {}
    for d in fpi_data:
        season = d['season']
        if season not in season_fpi:
            season_fpi[season] = {'pct': [], 'rank': []}
        if not np.isnan(d['fpi_pct_mean']):
            season_fpi[season]['pct'].append(d['fpi_pct_mean'])
        if not np.isnan(d['fpi_rank_mean']):
            season_fpi[season]['rank'].append(d['fpi_rank_mean'])

    # 计算每个赛季的平均 FPI
    seasons = sorted(season_fpi.keys())
    fpi_pct_means = []
    fpi_rank_means = []
    fpi_pct_stds = []
    fpi_rank_stds = []

    for s in seasons:
        pct_vals = season_fpi[s]['pct']
        rank_vals = season_fpi[s]['rank']
        fpi_pct_means.append(np.mean(pct_vals) if pct_vals else np.nan)
        fpi_rank_means.append(np.mean(rank_vals) if rank_vals else np.nan)
        fpi_pct_stds.append(np.std(pct_vals) if len(pct_vals) > 1 else 0)
        fpi_rank_stds.append(np.std(rank_vals) if len(rank_vals) > 1 else 0)

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(seasons))
    width = 0.35

    bars1 = ax.bar(x - width/2, fpi_pct_means, width, label='Percentage Method',
                   color='steelblue', alpha=0.7, yerr=fpi_pct_stds, capsize=3)
    bars2 = ax.bar(x + width/2, fpi_rank_means, width, label='Ranking Method',
                   color='coral', alpha=0.7, yerr=fpi_rank_stds, capsize=3)

    # 标注 S27 (Bobby Bones 赛季)
    if 27 in seasons:
        s27_idx = seasons.index(27)
        ax.annotate('S27\n(Bobby Bones)',
                    xy=(s27_idx, fpi_pct_means[s27_idx]),
                    xytext=(s27_idx, fpi_pct_means[s27_idx] + 0.15),
                    fontsize=10, ha='center', color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred'))

    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='FPI = 0.5')
    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Fan Power Index (FPI)', fontsize=12)
    ax.set_title('Fan Power Index by Season\n(Higher FPI = Fan votes dominate elimination)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(seasons, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_fpi_by_season.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_fpi_by_season.png")

    # 打印 S27 的特别分析
    if 27 in seasons:
        s27_idx = seasons.index(27)
        print(f"\n  S27 (Bobby Bones) Analysis:")
        print(f"    FPI (Percentage): {fpi_pct_means[s27_idx]:.3f}")
        print(f"    FPI (Ranking):    {fpi_rank_means[s27_idx]:.3f}")
        print(f"    Difference:       {fpi_pct_means[s27_idx] - fpi_rank_means[s27_idx]:.3f}")


def plot_fpi_scatter(fpi_data, output_dir):
    """
    散点图：百分比法 FPI vs 排名法 FPI
    每个点代表一周，颜色代表赛季
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(10, 10))

    # 收集数据
    fpi_pct = []
    fpi_rank = []
    seasons = []

    for d in fpi_data:
        if not np.isnan(d['fpi_pct_mean']) and not np.isnan(d['fpi_rank_mean']):
            fpi_pct.append(d['fpi_pct_mean'])
            fpi_rank.append(d['fpi_rank_mean'])
            seasons.append(d['season'])

    fpi_pct = np.array(fpi_pct)
    fpi_rank = np.array(fpi_rank)
    seasons = np.array(seasons)

    # 按赛季着色
    unique_seasons = sorted(set(seasons))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_seasons)))
    season_to_color = {s: colors[i] for i, s in enumerate(unique_seasons)}

    for s in unique_seasons:
        mask = seasons == s
        label = f'S{s}' if s == 27 else None  # 只标注 S27
        ax.scatter(fpi_pct[mask], fpi_rank[mask],
                   c=[season_to_color[s]], alpha=0.6, s=50, label=label)

    # 对角线
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y = x')

    # 0.5 参考线
    ax.axhline(y=0.5, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='red', linestyle=':', alpha=0.5)

    # 标注 S27 的点
    s27_mask = seasons == 27
    if s27_mask.any():
        ax.scatter(fpi_pct[s27_mask], fpi_rank[s27_mask],
                   c='red', s=100, marker='*', label='S27 (Bobby Bones)', zorder=10)

    ax.set_xlabel('FPI (Percentage Method)', fontsize=12)
    ax.set_ylabel('FPI (Ranking Method)', fontsize=12)
    ax.set_title('FPI Comparison: Percentage vs Ranking Method\n(Each point = one week)', fontsize=13)
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_fpi_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_fpi_scatter.png")


def plot_fpi_difference_by_week(fpi_data, output_dir):
    """
    展示 FPI 差异 (百分比法 - 排名法) 随周次的变化
    """
    setup_plot_style()

    # 按周分组
    week_diff = {}
    for d in fpi_data:
        week = d['week']
        if week not in week_diff:
            week_diff[week] = []
        diff = d['fpi_pct_mean'] - d['fpi_rank_mean']
        if not np.isnan(diff):
            week_diff[week].append(diff)

    weeks = sorted(week_diff.keys())
    diff_means = [np.mean(week_diff[w]) for w in weeks]
    diff_stds = [np.std(week_diff[w]) for w in weeks]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(weeks, diff_means, yerr=diff_stds, capsize=4,
           color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('FPI Difference (Percentage - Ranking)', fontsize=12)
    ax.set_title('FPI Method Difference by Week\n(Positive = Percentage method shows higher fan power)', fontsize=13)
    ax.set_xticks(weeks)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_fpi_diff_by_week.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_fpi_diff_by_week.png")


def find_high_fpi_weeks(fpi_data, threshold=0.7, n_top=10):
    """
    找到 FPI 特别高的周（争议周）
    """
    high_fpi_weeks = []
    for d in fpi_data:
        if d['fpi_pct_mean'] > threshold:
            high_fpi_weeks.append({
                'season': d['season'],
                'week': d['week'],
                'fpi_pct': d['fpi_pct_mean'],
                'fpi_rank': d['fpi_rank_mean'],
                'diff': d['fpi_pct_mean'] - d['fpi_rank_mean'],
                'n_contestants': d['n_contestants'],
            })

    # 按 FPI 排序
    high_fpi_weeks.sort(key=lambda x: -x['fpi_pct'])
    return high_fpi_weeks[:n_top]


def generate_fpi_summary_table(fpi_data, output_dir):
    """
    生成 FPI 汇总表格
    """
    # 按赛季分组
    season_stats = {}
    for d in fpi_data:
        season = d['season']
        if season not in season_stats:
            season_stats[season] = {'pct': [], 'rank': []}
        if not np.isnan(d['fpi_pct_mean']):
            season_stats[season]['pct'].append(d['fpi_pct_mean'])
        if not np.isnan(d['fpi_rank_mean']):
            season_stats[season]['rank'].append(d['fpi_rank_mean'])

    # 写入文件
    with open(f'{output_dir}/table_fpi_summary.txt', 'w') as f:
        f.write("Table: Fan Power Index (FPI) Summary by Season\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Season':<10} {'FPI (Pct)':<12} {'FPI (Rank)':<12} {'Difference':<12} {'N Weeks':<10} {'High FPI Weeks':<15}\n")
        f.write("-" * 90 + "\n")

        for season in sorted(season_stats.keys()):
            stats = season_stats[season]
            pct_mean = np.mean(stats['pct']) if stats['pct'] else np.nan
            rank_mean = np.mean(stats['rank']) if stats['rank'] else np.nan
            diff = pct_mean - rank_mean if not np.isnan(pct_mean) and not np.isnan(rank_mean) else np.nan
            n_weeks = len(stats['pct'])
            high_fpi_count = sum(1 for x in stats['pct'] if x > 0.5)

            marker = " ***" if season == 27 else ""
            f.write(f"{season:<10} {pct_mean:<12.3f} {rank_mean:<12.3f} {diff:<12.3f} {n_weeks:<10} {high_fpi_count:<15}{marker}\n")

        f.write("-" * 90 + "\n")

        # 总体统计
        all_pct = [d['fpi_pct_mean'] for d in fpi_data if not np.isnan(d['fpi_pct_mean'])]
        all_rank = [d['fpi_rank_mean'] for d in fpi_data if not np.isnan(d['fpi_rank_mean'])]
        f.write(f"{'Overall':<10} {np.mean(all_pct):<12.3f} {np.mean(all_rank):<12.3f} {np.mean(all_pct)-np.mean(all_rank):<12.3f} {len(all_pct):<10} {sum(1 for x in all_pct if x > 0.5):<15}\n")
        f.write("=" * 90 + "\n")
        f.write("\n*** S27 = Bobby Bones season (controversial winner)\n")
        f.write("High FPI Weeks = weeks where FPI (Percentage) > 0.5\n")

    print(f"  Saved: {output_dir}/table_fpi_summary.txt")


def print_fpi_summary(fpi_data):
    """打印 FPI 汇总统计"""
    print("\n" + "=" * 60)
    print("Fan Power Index (FPI) Summary")
    print("=" * 60)

    all_pct = [d['fpi_pct_mean'] for d in fpi_data if not np.isnan(d['fpi_pct_mean'])]
    all_rank = [d['fpi_rank_mean'] for d in fpi_data if not np.isnan(d['fpi_rank_mean'])]

    print(f"\nOverall Statistics:")
    print(f"  Total weeks analyzed: {len(all_pct)}")
    print(f"  FPI (Percentage Method): mean={np.mean(all_pct):.3f}, std={np.std(all_pct):.3f}")
    print(f"  FPI (Ranking Method):    mean={np.mean(all_rank):.3f}, std={np.std(all_rank):.3f}")
    print(f"  Average Difference:      {np.mean(all_pct) - np.mean(all_rank):.3f}")

    print(f"\nWeeks with FPI > 0.5 (fan-dominated):")
    print(f"  Percentage Method: {sum(1 for x in all_pct if x > 0.5)}/{len(all_pct)} ({100*sum(1 for x in all_pct if x > 0.5)/len(all_pct):.1f}%)")
    print(f"  Ranking Method:    {sum(1 for x in all_rank if x > 0.5)}/{len(all_rank)} ({100*sum(1 for x in all_rank if x > 0.5)/len(all_rank):.1f}%)")

    print(f"\nWeeks with FPI > 0.7 (strongly fan-dominated):")
    print(f"  Percentage Method: {sum(1 for x in all_pct if x > 0.7)}/{len(all_pct)} ({100*sum(1 for x in all_pct if x > 0.7)/len(all_pct):.1f}%)")
    print(f"  Ranking Method:    {sum(1 for x in all_rank if x > 0.7)}/{len(all_rank)} ({100*sum(1 for x in all_rank if x > 0.7)/len(all_rank):.1f}%)")

    # 找到高 FPI 周
    high_fpi_weeks = find_high_fpi_weeks(fpi_data, threshold=0.7, n_top=5)
    if high_fpi_weeks:
        print(f"\nTop 5 High FPI Weeks (Percentage Method > 0.7):")
        for hw in high_fpi_weeks:
            print(f"  Season {hw['season']}, Week {hw['week']}: FPI_pct={hw['fpi_pct']:.3f}, FPI_rank={hw['fpi_rank']:.3f}, diff={hw['diff']:.3f}")

    print("=" * 60)


def compute_variance_decomposition(datas):
    """
    计算每周的方差分解

    返回:
        list of dict: 每周的方差分解数据
    """
    P_fan_samples = datas['P_fan_samples']
    td = datas['train_data']
    week_data = td['week_data']

    results = []
    P_fan_mean = P_fan_samples.mean(axis=0)

    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        n_contestants = len(indices)

        if n_contestants < 2:
            continue

        season = wd['season']
        week = wd['week']

        judge_pct = td['judge_score_pct'][mask]
        judge_rank = td['judge_rank_score'][mask]
        P_fan = P_fan_mean[mask]

        # === 百分比法 ===
        var_judge_pct = np.var(judge_pct)
        var_fan_pct = np.var(P_fan)
        var_total_pct = np.var(judge_pct + P_fan)

        # === 排名法 ===
        diff_mat = P_fan[:, None] - P_fan[None, :]
        soft_rank = np.sum(1 / (1 + np.exp(-diff_mat / 0.1)), axis=1) - 0.5
        R_fan = soft_rank / (n_contestants - 1) if n_contestants > 1 else np.array([0.5])

        var_judge_rank = np.var(judge_rank)
        var_fan_rank = np.var(R_fan)
        var_total_rank = np.var(judge_rank + R_fan)

        results.append({
            'season': season,
            'week': week,
            'n_contestants': n_contestants,
            'var_judge_pct': var_judge_pct,
            'var_fan_pct': var_fan_pct,
            'var_total_pct': var_total_pct,
            'var_judge_rank': var_judge_rank,
            'var_fan_rank': var_fan_rank,
            'var_total_rank': var_total_rank,
        })

    return results


def plot_power_shift(var_data, output_dir):
    """
    影响力演变图 (Power Shift Plot)
    堆叠面积图展示评委分和粉丝分的方差占比
    """
    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 按赛季分组
    season_var = {}
    for d in var_data:
        s = d['season']
        if s not in season_var:
            season_var[s] = {'judge_pct': [], 'fan_pct': [], 'judge_rank': [], 'fan_rank': []}

        # 归一化
        total_pct = d['var_judge_pct'] + d['var_fan_pct']
        total_rank = d['var_judge_rank'] + d['var_fan_rank']

        if total_pct > 1e-10:
            season_var[s]['judge_pct'].append(d['var_judge_pct'] / total_pct)
            season_var[s]['fan_pct'].append(d['var_fan_pct'] / total_pct)
        if total_rank > 1e-10:
            season_var[s]['judge_rank'].append(d['var_judge_rank'] / total_rank)
            season_var[s]['fan_rank'].append(d['var_fan_rank'] / total_rank)

    seasons = sorted(season_var.keys())
    judge_pct_avg = [np.mean(season_var[s]['judge_pct']) if season_var[s]['judge_pct'] else 0.5 for s in seasons]
    fan_pct_avg = [np.mean(season_var[s]['fan_pct']) if season_var[s]['fan_pct'] else 0.5 for s in seasons]
    judge_rank_avg = [np.mean(season_var[s]['judge_rank']) if season_var[s]['judge_rank'] else 0.5 for s in seasons]
    fan_rank_avg = [np.mean(season_var[s]['fan_rank']) if season_var[s]['fan_rank'] else 0.5 for s in seasons]

    # 左图：百分比法
    ax1 = axes[0]
    ax1.stackplot(seasons, [judge_pct_avg, fan_pct_avg],
                  labels=['Judge Variance', 'Fan Variance'],
                  colors=['#3498DB', '#E74C3C'], alpha=0.8)
    ax1.axhline(y=0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Season', fontsize=12)
    ax1.set_ylabel('Variance Share', fontsize=12)
    ax1.set_title('Percentage Method\n(Fan power can dominate)', fontsize=13)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(seasons[0], seasons[-1])

    # 右图：排名法
    ax2 = axes[1]
    ax2.stackplot(seasons, [judge_rank_avg, fan_rank_avg],
                  labels=['Judge Variance', 'Fan Variance'],
                  colors=['#3498DB', '#E74C3C'], alpha=0.8)
    ax2.axhline(y=0.5, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Season', fontsize=12)
    ax2.set_ylabel('Variance Share', fontsize=12)
    ax2.set_title('Ranking Method\n(More balanced power)', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(seasons[0], seasons[-1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_power_shift.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_power_shift.png")
