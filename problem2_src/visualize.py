"""
第二问可视化函数
比较排名法和百分比法的差异
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 争议周阈值：与 filter_posterior.py 保持一致
CONTROVERSIAL_THRESHOLD = 0.01  # 1%


def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150


def get_controversial_weeks(datas):
    """
    获取争议周信息

    返回:
        controversial_set: set of (season, week) tuples for controversial weeks
        week_filter_info: list of week info dicts (if available)
    """
    week_filter_info = datas.get('week_filter_info', [])
    controversial_set = set()

    for info in week_filter_info:
        # 使用阈值判断，兼容旧数据
        is_controversial = info.get('controversial', False)
        if not is_controversial and 'rate' in info:
            is_controversial = info['rate'] < CONTROVERSIAL_THRESHOLD
        if is_controversial:
            controversial_set.add((info['season'], info['week']))

    return controversial_set, week_filter_info


def compute_scores_both_methods(datas):
    """
    对每个观测点，分别用排名法和百分比法计算 S 分数
    使用筛选后的 P_fan 均值

    返回:
        S_pct_mean: [n_obs] 百分比法得分均值
        S_rank_mean: [n_obs] 排名法得分均值
    """
    # 优先使用筛选后的数据
    if 'pfan_filtered' in datas:
        P_fan_mean = datas['pfan_filtered']['mean']
        print("  Using filtered P_fan data")
    else:
        print("  Warning: pfan_filtered not found, using raw P_fan_samples mean")
        P_fan_mean = datas['P_fan_samples'].mean(axis=0)

    td = datas['train_data']
    week_data = td['week_data']
    n_obs = len(P_fan_mean)

    S_pct_mean = np.zeros(n_obs, dtype=np.float32)
    S_rank_mean = np.zeros(n_obs, dtype=np.float32)

    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        n_contestants = len(indices)

        # 百分比法: S = judge_pct + P_fan
        S_pct_mean[mask] = td['judge_score_pct'][mask] + P_fan_mean[mask]

        # 排名法: S = judge_rank + R_fan
        P_week = P_fan_mean[mask]
        if n_contestants > 1:
            diff = P_week[:, None] - P_week[None, :]
            soft_rank = np.sum(1 / (1 + np.exp(-diff / 0.1)), axis=1) - 0.5
            R_fan = soft_rank / (n_contestants - 1)
        else:
            R_fan = np.array([0.5], dtype=np.float32)
        S_rank_mean[mask] = td['judge_rank_score'][mask] + R_fan

    return S_pct_mean, S_rank_mean


def compute_rule_comparison_metrics(datas, S_pct_mean, S_rank_mean):
    """
    计算两种规则的比较指标
    使用筛选后的得分均值

    返回:
        metrics: dict 包含各种比较指标
        week_details: list 每周的详细比较结果
    """
    td = datas['train_data']
    week_data = td['week_data']

    # 获取争议周信息
    controversial_set, _ = get_controversial_weeks(datas)

    week_details = []
    total_rank_diff = 0
    total_contestants = 0
    consistent_weeks = 0
    total_elim_weeks = 0
    n_controversial = 0

    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        n_contestants = len(indices)
        n_elim = wd['n_eliminated']
        season = wd['season']
        week = wd['week']

        if n_contestants < 2:
            continue

        # 检查是否为争议周
        is_controversial = (season, week) in controversial_set
        if is_controversial:
            n_controversial += 1

        # 使用筛选后的均值（已经是均值了，不需要再取均值）
        S_pct_week = S_pct_mean[mask]
        S_rank_week = S_rank_mean[mask]

        # 计算排名
        rank_pct = np.argsort(np.argsort(-S_pct_week))  # 高分排前面
        rank_rank = np.argsort(np.argsort(-S_rank_week))

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
            pred_elim_pct = set(np.argsort(S_pct_week)[:n_elim])
            pred_elim_rank = set(np.argsort(S_rank_week)[:n_elim])
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
            'S_pct_mean': S_pct_week,
            'S_rank_mean': S_rank_week,
            'obs_indices': indices,
            'is_controversial': is_controversial,
        })

    metrics = {
        'avg_rank_volatility': total_rank_diff / total_contestants if total_contestants > 0 else 0,
        'elim_consistency': consistent_weeks / total_elim_weeks if total_elim_weeks > 0 else 0,
        'total_weeks': len(week_details),
        'total_elim_weeks': total_elim_weeks,
        'consistent_weeks': consistent_weeks,
        'n_controversial': n_controversial,
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
    print(f"争议周数: {metrics.get('n_controversial', 0)}")
    print("=" * 60)


def generate_summary_table(metrics, week_details, output_dir):
    """生成汇总表格，包含争议周统计"""
    # 按赛季分组统计
    season_stats = {}
    for wd in week_details:
        season = wd['season']
        if season not in season_stats:
            season_stats[season] = {
                'rank_diffs': [],
                'elim_consistent': [],
                'n_weeks': 0,
                'n_controversial': 0,
            }
        season_stats[season]['rank_diffs'].append(wd['avg_rank_diff'])
        if wd['elim_consistent'] is not None:
            season_stats[season]['elim_consistent'].append(wd['elim_consistent'])
        season_stats[season]['n_weeks'] += 1
        if wd.get('is_controversial', False):
            season_stats[season]['n_controversial'] += 1

    # 写入文件
    with open(f'{output_dir}/table_rule_comparison.txt', 'w') as f:
        f.write("Table: Rule Comparison Summary by Season\n")
        f.write("=" * 95 + "\n")
        f.write(f"{'Season':<10} {'Avg Rank Diff':<15} {'Elim Consistency':<18} {'N Weeks':<10} {'Controversial':<12}\n")
        f.write("-" * 95 + "\n")

        for season in sorted(season_stats.keys()):
            stats = season_stats[season]
            avg_diff = np.mean(stats['rank_diffs'])
            if stats['elim_consistent']:
                consistency = np.mean(stats['elim_consistent'])
                consistency_str = f"{consistency:.1%}"
            else:
                consistency_str = "N/A"
            f.write(f"{season:<10} {avg_diff:<15.2f} {consistency_str:<18} {stats['n_weeks']:<10} {stats['n_controversial']:<12}\n")

        f.write("-" * 95 + "\n")
        consistency_str = f"{metrics['elim_consistency']:.1%}"
        f.write(f"{'Overall':<10} {metrics['avg_rank_volatility']:<15.2f} "
                f"{consistency_str:<18} {metrics['total_weeks']:<10} {metrics.get('n_controversial', 0):<12}\n")
        f.write("=" * 95 + "\n")

    print(f"  Saved: {output_dir}/table_rule_comparison.txt")

    # 打印到控制台
    print("\n  Table: Rule Comparison Summary by Season")
    print("  " + "=" * 80)
    print(f"  {'Season':<8} {'Avg Rank Diff':<14} {'Elim Consistency':<16} {'N Weeks':<10} {'Controv.':<8}")
    print("  " + "-" * 80)
    for season in sorted(season_stats.keys())[:10]:  # 只打印前10个赛季
        stats = season_stats[season]
        avg_diff = np.mean(stats['rank_diffs'])
        if stats['elim_consistent']:
            consistency = np.mean(stats['elim_consistent'])
            consistency_str = f"{consistency:.1%}"
        else:
            consistency_str = "N/A"
        print(f"  {season:<8} {avg_diff:<14.2f} {consistency_str:<16} {stats['n_weeks']:<10} {stats['n_controversial']:<8}")
    if len(season_stats) > 10:
        print(f"  ... ({len(season_stats) - 10} more seasons)")
    print("  " + "-" * 80)
    consistency_str = f"{metrics['elim_consistency']:.1%}"
    print(f"  {'Overall':<8} {metrics['avg_rank_volatility']:<14.2f} "
          f"{consistency_str:<16} {metrics['total_weeks']:<10} {metrics.get('n_controversial', 0):<8}")
    print("  " + "=" * 70)


def plot_rule_heatmap(week_details, output_dir):
    """
    规则偏差热力图
    X轴为赛季，Y轴为周次，颜色深浅代表排名差异程度
    争议周用特殊标记
    """
    setup_plot_style()

    # 收集数据
    seasons = sorted(set(wd['season'] for wd in week_details))
    weeks = sorted(set(wd['week'] for wd in week_details))

    # 创建矩阵
    heatmap_data = np.full((len(weeks), len(seasons)), np.nan)
    controversial_mask = np.zeros((len(weeks), len(seasons)), dtype=bool)
    season_to_idx = {s: i for i, s in enumerate(seasons)}
    week_to_idx = {w: i for i, w in enumerate(weeks)}

    for wd in week_details:
        si = season_to_idx[wd['season']]
        wi = week_to_idx[wd['week']]
        heatmap_data[wi, si] = wd['avg_rank_diff']
        if wd.get('is_controversial', False):
            controversial_mask[wi, si] = True

    # 绘图
    fig, ax = plt.subplots(figsize=(14, 8))

    # 使用 masked array 处理 NaN
    masked_data = np.ma.masked_invalid(heatmap_data)

    im = ax.imshow(masked_data, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest')

    # 标记争议周（用黑色边框）
    for wi in range(len(weeks)):
        for si in range(len(seasons)):
            if controversial_mask[wi, si]:
                rect = plt.Rectangle((si - 0.5, wi - 0.5), 1, 1,
                                      fill=False, edgecolor='black',
                                      linewidth=2, linestyle='--')
                ax.add_patch(rect)

    # 设置坐标轴
    ax.set_xticks(np.arange(0, len(seasons), max(1, len(seasons)//20)))
    ax.set_xticklabels([seasons[i] for i in range(0, len(seasons), max(1, len(seasons)//20))],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(weeks)))
    ax.set_yticklabels(weeks, fontsize=9)

    ax.set_xlabel('Season', fontsize=12)
    ax.set_ylabel('Week', fontsize=12)
    ax.set_title('Rule Deviation Heatmap\n(Average Rank Difference: Percentage vs Ranking Method)\nDashed boxes = Controversial weeks',
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
    争议周用垂直虚线标记
    """
    setup_plot_style()

    td = datas['train_data']
    celeb_idx = td['celeb_idx']

    # 构建 (season, week) -> is_controversial 映射
    controv_map = {(wd['season'], wd['week']): wd.get('is_controversial', False)
                   for wd in week_details}

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
                'is_controversial': wd.get('is_controversial', False),
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
    is_controversial = [d['is_controversial'] for d in season_data]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(weeks, ranks_pct, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Percentage Method')
    ax.plot(weeks, ranks_rank, 's--', color='coral', linewidth=2, markersize=8,
            label='Ranking Method')

    # 标记争议周
    for i, (w, controv) in enumerate(zip(weeks, is_controversial)):
        if controv:
            ax.axvline(x=w, color='orange', linestyle=':', alpha=0.7, linewidth=2)

    # 反转 Y 轴（排名1在上面）
    ax.invert_yaxis()

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Rank (1 = Best)', fontsize=12)

    # 检查是否有争议周
    n_controv = sum(is_controversial)
    title_suffix = f'\n(Orange lines = {n_controv} controversial weeks)' if n_controv > 0 else ''
    ax.set_title(f'Rank Trajectory Comparison: Celebrity {celeb_id} (Season {target_season}){title_suffix}',
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

