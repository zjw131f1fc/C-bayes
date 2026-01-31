"""
第二问可视化函数
比较排名法和百分比法的差异
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150


def compute_scores_both_methods(datas):
    """
    对每个观测点，分别用排名法和百分比法计算 S 分数

    返回:
        S_pct: [n_samples, n_obs] 百分比法得分
        S_rank: [n_samples, n_obs] 排名法得分
    """
    P_fan_samples = datas['P_fan_samples']  # [n_samples, n_obs]
    td = datas['train_data']
    week_data = td['week_data']
    n_samples, n_obs = P_fan_samples.shape

    S_pct = np.zeros((n_samples, n_obs), dtype=np.float32)
    S_rank = np.zeros((n_samples, n_obs), dtype=np.float32)

    for s in range(n_samples):
        P_fan = P_fan_samples[s]

        for wd in week_data:
            mask = wd['obs_mask']
            indices = np.where(mask)[0]
            n_contestants = len(indices)

            # 百分比法: S = judge_pct + P_fan
            S_pct[s, mask] = td['judge_score_pct'][mask] + P_fan[mask]

            # 排名法: S = judge_rank + R_fan
            P_week = P_fan[mask]
            if n_contestants > 1:
                diff = P_week[:, None] - P_week[None, :]
                soft_rank = np.sum(1 / (1 + np.exp(-diff / 0.1)), axis=1) - 0.5
                R_fan = soft_rank / (n_contestants - 1)
            else:
                R_fan = np.array([0.5], dtype=np.float32)
            S_rank[s, mask] = td['judge_rank_score'][mask] + R_fan

    return S_pct, S_rank


def compute_rule_comparison_metrics(datas, S_pct, S_rank):
    """
    计算两种规则的比较指标

    返回:
        metrics: dict 包含各种比较指标
        week_details: list 每周的详细比较结果
    """
    td = datas['train_data']
    week_data = td['week_data']
    n_samples = S_pct.shape[0]

    week_details = []
    total_rank_diff = 0
    total_contestants = 0
    consistent_weeks = 0
    total_elim_weeks = 0

    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        n_contestants = len(indices)
        n_elim = wd['n_eliminated']
        season = wd['season']
        week = wd['week']

        if n_contestants < 2:
            continue

        # 计算后验均值
        S_pct_mean = S_pct[:, mask].mean(axis=0)
        S_rank_mean = S_rank[:, mask].mean(axis=0)

        # 计算排名
        rank_pct = np.argsort(np.argsort(-S_pct_mean))  # 高分排前面
        rank_rank = np.argsort(np.argsort(-S_rank_mean))

        # 排名差异
        rank_diff = np.abs(rank_pct - rank_rank)
        avg_rank_diff = np.mean(rank_diff)
        total_rank_diff += np.sum(rank_diff)
        total_contestants += n_contestants

        # 淘汰一致性
        elim_consistent = None
        if n_elim > 0:
            total_elim_weeks += 1
            # 预测淘汰者（得分最低的 n_elim 人）
            pred_elim_pct = set(np.argsort(S_pct_mean)[:n_elim])
            pred_elim_rank = set(np.argsort(S_rank_mean)[:n_elim])
            elim_consistent = pred_elim_pct == pred_elim_rank
            if elim_consistent:
                consistent_weeks += 1

        week_details.append({
            'season': season,
            'week': week,
            'n_contestants': n_contestants,
            'n_eliminated': n_elim,
            'avg_rank_diff': avg_rank_diff,
            'max_rank_diff': np.max(rank_diff),
            'elim_consistent': elim_consistent,
            'rank_pct': rank_pct,
            'rank_rank': rank_rank,
            'S_pct_mean': S_pct_mean,
            'S_rank_mean': S_rank_mean,
            'obs_indices': indices,
        })

    metrics = {
        'avg_rank_volatility': total_rank_diff / total_contestants if total_contestants > 0 else 0,
        'elim_consistency': consistent_weeks / total_elim_weeks if total_elim_weeks > 0 else 0,
        'total_weeks': len(week_details),
        'total_elim_weeks': total_elim_weeks,
        'consistent_weeks': consistent_weeks,
    }

    return metrics, week_details


def print_summary_stats(datas, metrics):
    """打印第二问需要的汇总统计数据"""
    print("\n" + "=" * 60)
    print("第二问 - 规则比较汇总统计")
    print("=" * 60)
    print(f"\n排名变动率 (Rank Volatility):")
    print(f"  平均排名差异: {metrics['avg_rank_volatility']:.2f}")
    print(f"\n结果一致性:")
    print(f"  淘汰一致周数: {metrics['consistent_weeks']}/{metrics['total_elim_weeks']}")
    print(f"  一致性比例: {metrics['elim_consistency']:.1%}")
    print(f"\n总周数: {metrics['total_weeks']}")
    print("=" * 60)


def generate_summary_table(metrics, week_details, output_dir):
    """生成汇总表格"""
    # 按赛季分组统计
    season_stats = {}
    for wd in week_details:
        season = wd['season']
        if season not in season_stats:
            season_stats[season] = {
                'rank_diffs': [],
                'elim_consistent': [],
                'n_weeks': 0,
            }
        season_stats[season]['rank_diffs'].append(wd['avg_rank_diff'])
        if wd['elim_consistent'] is not None:
            season_stats[season]['elim_consistent'].append(wd['elim_consistent'])
        season_stats[season]['n_weeks'] += 1

    # 写入文件
    with open(f'{output_dir}/table_rule_comparison.txt', 'w') as f:
        f.write("Table: Rule Comparison Summary by Season\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Season':<10} {'Avg Rank Diff':<15} {'Elim Consistency':<18} {'N Weeks':<10}\n")
        f.write("-" * 80 + "\n")

        for season in sorted(season_stats.keys()):
            stats = season_stats[season]
            avg_diff = np.mean(stats['rank_diffs'])
            if stats['elim_consistent']:
                consistency = np.mean(stats['elim_consistent'])
                consistency_str = f"{consistency:.1%}"
            else:
                consistency_str = "N/A"
            f.write(f"{season:<10} {avg_diff:<15.2f} {consistency_str:<18} {stats['n_weeks']:<10}\n")

        f.write("-" * 80 + "\n")
        consistency_str = f"{metrics['elim_consistency']:.1%}"
        f.write(f"{'Overall':<10} {metrics['avg_rank_volatility']:<15.2f} "
                f"{consistency_str:<18} {metrics['total_weeks']:<10}\n")
        f.write("=" * 80 + "\n")

    print(f"  Saved: {output_dir}/table_rule_comparison.txt")

    # 打印到控制台
    print("\n  Table: Rule Comparison Summary by Season")
    print("  " + "=" * 70)
    print(f"  {'Season':<10} {'Avg Rank Diff':<15} {'Elim Consistency':<18} {'N Weeks':<10}")
    print("  " + "-" * 70)
    for season in sorted(season_stats.keys())[:10]:  # 只打印前10个赛季
        stats = season_stats[season]
        avg_diff = np.mean(stats['rank_diffs'])
        if stats['elim_consistent']:
            consistency = np.mean(stats['elim_consistent'])
            consistency_str = f"{consistency:.1%}"
        else:
            consistency_str = "N/A"
        print(f"  {season:<10} {avg_diff:<15.2f} {consistency_str:<18} {stats['n_weeks']:<10}")
    if len(season_stats) > 10:
        print(f"  ... ({len(season_stats) - 10} more seasons)")
    print("  " + "-" * 70)
    consistency_str = f"{metrics['elim_consistency']:.1%}"
    print(f"  {'Overall':<10} {metrics['avg_rank_volatility']:<15.2f} "
          f"{consistency_str:<18} {metrics['total_weeks']:<10}")
    print("  " + "=" * 70)


def plot_rule_heatmap(week_details, output_dir):
    """
    规则偏差热力图
    X轴为赛季，Y轴为周次，颜色深浅代表排名差异程度
    """
    setup_plot_style()

    # 收集数据
    seasons = sorted(set(wd['season'] for wd in week_details))
    weeks = sorted(set(wd['week'] for wd in week_details))

    # 创建矩阵
    heatmap_data = np.full((len(weeks), len(seasons)), np.nan)
    season_to_idx = {s: i for i, s in enumerate(seasons)}
    week_to_idx = {w: i for i, w in enumerate(weeks)}

    for wd in week_details:
        si = season_to_idx[wd['season']]
        wi = week_to_idx[wd['week']]
        heatmap_data[wi, si] = wd['avg_rank_diff']

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 使用 masked array 处理 NaN
    masked_data = np.ma.masked_invalid(heatmap_data)

    im = ax.imshow(masked_data, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    # 设置坐标轴
    ax.set_xticks(np.arange(0, len(seasons), max(1, len(seasons)//20)))
    ax.set_xticklabels([seasons[i] for i in range(0, len(seasons), max(1, len(seasons)//20))],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(weeks)))
    ax.set_yticklabels(weeks, fontsize=9)

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Week', fontsize=12)
    ax.set_title('Rule Deviation Heatmap\n(Average Rank Difference: Percentage vs Ranking Method)',
                 fontsize=13)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Rank Difference', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_rule_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_rule_heatmap.png")


def plot_bump_chart(datas, week_details, celeb_id, output_dir):
    """
    淘汰轨迹对比图 (Bump Chart)
    展示某位选手在两种规则下的排名演变差异
    """
    setup_plot_style()

    td = datas['train_data']
    celeb_idx = td['celeb_idx']

    # 找到该选手参与的周
    celeb_weeks = []
    for wd in week_details:
        obs_indices = wd['obs_indices']
        celeb_mask = celeb_idx[obs_indices] == celeb_id
        if celeb_mask.any():
            local_idx = np.where(celeb_mask)[0][0]
            celeb_weeks.append({
                'season': wd['season'],
                'week': wd['week'],
                'rank_pct': wd['rank_pct'][local_idx] + 1,  # 1-indexed
                'rank_rank': wd['rank_rank'][local_idx] + 1,
                'n_contestants': wd['n_contestants'],
            })

    if not celeb_weeks:
        print(f"  Warning: No data found for celeb_id={celeb_id}")
        return

    # 按赛季分组
    seasons = {}
    for cw in celeb_weeks:
        s = cw['season']
        if s not in seasons:
            seasons[s] = []
        seasons[s].append(cw)

    # 选择周数最多的赛季
    target_season = max(seasons.keys(), key=lambda s: len(seasons[s]))
    season_data = sorted(seasons[target_season], key=lambda x: x['week'])

    weeks = [d['week'] for d in season_data]
    ranks_pct = [d['rank_pct'] for d in season_data]
    ranks_rank = [d['rank_rank'] for d in season_data]
    n_contestants = [d['n_contestants'] for d in season_data]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(weeks, ranks_pct, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Percentage Method')
    ax.plot(weeks, ranks_rank, 's--', color='coral', linewidth=2, markersize=8,
            label='Ranking Method')

    # 反转 Y 轴（排名1在上面）
    ax.invert_yaxis()

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Rank (1 = Best)', fontsize=12)
    ax.set_title(f'Rank Trajectory Comparison: Celebrity {celeb_id} (Season {target_season})',
                 fontsize=13)
    ax.legend(loc='best')
    ax.set_xticks(weeks)
    ax.grid(True, alpha=0.3)

    # 添加选手数量标注
    for i, (w, n) in enumerate(zip(weeks, n_contestants)):
        ax.annotate(f'n={n}', (w, max(ranks_pct[i], ranks_rank[i]) + 0.5),
                    fontsize=8, ha='center', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_bump_chart_celeb_{celeb_id}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_bump_chart_celeb_{celeb_id}.png")


def find_interesting_celebs(datas, week_details, n_celebs=5):
    """
    找到排名差异最大的选手（适合做 Bump Chart）
    """
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    n_total_celebs = td['n_celebs']

    celeb_diffs = []
    for c in range(n_total_celebs):
        total_diff = 0
        count = 0
        for wd in week_details:
            obs_indices = wd['obs_indices']
            celeb_mask = celeb_idx[obs_indices] == c
            if celeb_mask.any():
                local_idx = np.where(celeb_mask)[0][0]
                diff = abs(wd['rank_pct'][local_idx] - wd['rank_rank'][local_idx])
                total_diff += diff
                count += 1

        if count >= 5:  # 至少参与5周
            celeb_diffs.append((c, total_diff / count, count))

    # 按平均差异排序
    celeb_diffs.sort(key=lambda x: -x[1])
    return celeb_diffs[:n_celebs]

