"""
争议选手分析和可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def setup_plot_style():
    """设置论文级绘图样式"""
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


def compute_rankings_both_methods(datas, celeb_id, target_season):
    """
    计算某选手在指定赛季两种规则下的排名

    返回:
        list of dict: 每周的排名数据
    """
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    week_data = td['week_data']

    # 优先使用筛选后的后验数据
    if 'pfan_filtered' in datas and 'mean' in datas['pfan_filtered']:
        P_fan_mean = datas['pfan_filtered']['mean']
    else:
        # 回退到使用 P_fan_samples 的均值
        P_fan_samples = datas['P_fan_samples']
        P_fan_mean = P_fan_samples.mean(axis=0)

    results = []

    for wd in week_data:
        if wd['season'] != target_season:
            continue

        mask = wd['obs_mask']
        indices = np.where(mask)[0]

        celeb_in_week = celeb_idx[indices] == celeb_id
        if not celeb_in_week.any():
            continue

        local_idx = np.where(celeb_in_week)[0][0]
        obs_idx = indices[local_idx]
        n_contestants = len(indices)

        P_fan_week = P_fan_mean[mask]
        judge_pct = td['judge_score_pct'][mask]
        judge_rank = td['judge_rank_score'][mask]

        # 百分比法
        S_pct = judge_pct + P_fan_week
        rank_pct = np.argsort(np.argsort(-S_pct))[local_idx] + 1

        # 排名法
        diff_mat = P_fan_week[:, None] - P_fan_week[None, :]
        soft_rank = np.sum(1 / (1 + np.exp(-diff_mat / 0.1)), axis=1) - 0.5
        R_fan = soft_rank / (n_contestants - 1) if n_contestants > 1 else np.array([0.5])
        S_rank = judge_rank + R_fan
        rank_rank = np.argsort(np.argsort(-S_rank))[local_idx] + 1

        results.append({
            'week': wd['week'],
            'n_contestants': n_contestants,
            'rank_pct': rank_pct,
            'rank_rank': rank_rank,
            'rank_diff': rank_pct - rank_rank,
            'eliminated': wd['eliminated_mask'][obs_idx],
            'judge_pct': judge_pct[local_idx],
            'judge_rank': judge_rank[local_idx],
            'P_fan': P_fan_week[local_idx],
        })

    return sorted(results, key=lambda x: x['week'])


def plot_bump_chart_single(datas, celeb_id, celeb_name, target_season,
                           actual_season, output_dir):
    """
    为单个选手绘制排名轨迹对比图 (Bump Chart)

    参数:
        celeb_id: 选手ID
        celeb_name: 选手名称
        target_season: 数据中的赛季编号
        actual_season: 实际赛季编号（用于标题）
        output_dir: 输出目录
    """
    setup_plot_style()

    results = compute_rankings_both_methods(datas, celeb_id, target_season)

    if not results:
        print(f"  Warning: No data found for {celeb_name}")
        return

    weeks = [r['week'] for r in results]
    ranks_pct = [r['rank_pct'] for r in results]
    ranks_rank = [r['rank_rank'] for r in results]
    n_contestants = [r['n_contestants'] for r in results]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制两条排名线
    ax.plot(weeks, ranks_pct, 'o-', color='#E74C3C', linewidth=2.5, markersize=10,
            label='Percentage Method', zorder=3)
    ax.plot(weeks, ranks_rank, 's--', color='#3498DB', linewidth=2.5, markersize=10,
            label='Ranking Method', zorder=3)

    # 填充两条线之间的区域（高亮差异）
    ax.fill_between(weeks, ranks_pct, ranks_rank, alpha=0.2, color='#9B59B6',
                    label='Rank Difference')

    # 反转Y轴（排名1在上面）
    ax.invert_yaxis()

    # 设置Y轴范围（留出边距）
    max_rank = max(max(ranks_pct), max(ranks_rank))
    ax.set_ylim(max_rank + 0.5, 0.5)

    # 添加选手数量标注
    for i, (w, n) in enumerate(zip(weeks, n_contestants)):
        y_pos = max(ranks_pct[i], ranks_rank[i]) + 0.3
        ax.annotate(f'n={n}', (w, y_pos), fontsize=9, ha='center',
                    alpha=0.7, color='gray')

    # 标注淘汰点
    for r in results:
        if r['eliminated']:
            ax.scatter([r['week']], [r['rank_pct']], marker='X', s=200,
                       color='darkred', zorder=5, label='Eliminated')
            break

    # 添加水平参考线
    ax.axhline(y=1, color='gold', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.axhline(y=3, color='silver', linestyle=':', alpha=0.5, linewidth=1.5)

    # 设置标签
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Rank (1 = Best)', fontsize=12)
    ax.set_title(f'Rank Trajectory Comparison: {celeb_name}\n(Season {actual_season})',
                 fontsize=14, fontweight='bold')

    # 设置X轴刻度
    ax.set_xticks(weeks)
    ax.set_xticklabels([f'W{w}' for w in weeks])

    # 设置Y轴刻度为整数
    ax.set_yticks(range(1, max_rank + 1))

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # 图例
    ax.legend(loc='lower right', framealpha=0.9)

    # 添加统计信息文本框
    avg_diff = np.mean([r['rank_diff'] for r in results])
    max_diff = min([r['rank_diff'] for r in results])  # 负值表示百分比法更有利
    textstr = f'Avg Rank Diff: {avg_diff:+.1f}\nMax Advantage: {max_diff:+d}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()

    # 保存
    filename = f'fig_bump_{celeb_name.replace(" ", "_")}_S{actual_season}.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved: {output_dir}/{filename}")


def plot_rank_diff_bar(datas, celeb_id, celeb_name, target_season,
                       actual_season, output_dir):
    """
    绘制排名差异柱状图
    """
    setup_plot_style()

    results = compute_rankings_both_methods(datas, celeb_id, target_season)

    if not results:
        return

    weeks = [r['week'] for r in results]
    diffs = [r['rank_diff'] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#E74C3C' if d < 0 else '#3498DB' for d in diffs]
    bars = ax.bar(weeks, diffs, color=colors, edgecolor='black', linewidth=0.5)

    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Rank Difference (Pct - Rank)', fontsize=12)
    ax.set_title(f'Rank Difference by Week: {celeb_name} (S{actual_season})\n'
                 f'Negative = Percentage Method Favors', fontsize=13)
    ax.set_xticks(weeks)
    ax.set_xticklabels([f'W{w}' for w in weeks])
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, diff in zip(bars, diffs):
        height = bar.get_height()
        ax.annotate(f'{diff:+d}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -10),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    filename = f'fig_diff_{celeb_name.replace(" ", "_")}_S{actual_season}.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{filename}")


def simulate_season_4scenarios(datas, target_season, n_simulations=1000):
    """
    模拟一个赛季在4种场景下的淘汰结果

    场景:
        A: 排名法 + 无评委拯救
        B: 百分比法 + 无评委拯救
        C: 排名法 + 有评委拯救
        D: 百分比法 + 有评委拯救

    返回:
        dict: 每个选手在每种场景下的淘汰周次分布

    注意:
        此函数模拟整个赛季的淘汰链，需要使用原始 P_fan_samples 来估计不确定性。
        不使用 pfan_filtered，因为筛选是按周独立进行的，不适合模拟整个赛季。
    """
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    week_data = td['week_data']

    # 使用原始 P_fan_samples 进行模拟（不使用筛选后的数据）
    # 因为筛选是按周独立进行的，不适合模拟整个赛季的淘汰链
    P_fan_samples = datas['P_fan_samples']

    # 获取该赛季的所有周数据
    season_weeks = [wd for wd in week_data if wd['season'] == target_season]
    season_weeks = sorted(season_weeks, key=lambda x: x['week'])

    if not season_weeks:
        return None

    # 获取该赛季所有选手
    all_celebs = set()
    for wd in season_weeks:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        all_celebs.update(celeb_idx[indices])

    # 初始化结果
    results = {cid: {'A': [], 'B': [], 'C': [], 'D': []} for cid in all_celebs}

    n_samples = min(n_simulations, P_fan_samples.shape[0])

    for sim in range(n_samples):
        P_fan = P_fan_samples[sim]

        # 对每种场景模拟
        for scenario in ['A', 'B', 'C', 'D']:
            use_pct = scenario in ['B', 'D']
            use_save = scenario in ['C', 'D']

            # 已淘汰选手集合
            eliminated = set()
            elimination_week = {cid: None for cid in all_celebs}

            for wd in season_weeks:
                mask = wd['obs_mask']
                indices = np.where(mask)[0]
                week_celebs = [celeb_idx[i] for i in indices]

                # 本周参与且未被淘汰的选手
                active_indices = []
                active_celebs_list = []
                for i, cid in zip(indices, week_celebs):
                    if cid not in eliminated:
                        active_indices.append(i)
                        active_celebs_list.append(cid)

                n_active = len(active_indices)
                if n_active <= 1:
                    continue

                n_elim = wd['n_eliminated']
                if n_elim == 0:
                    continue

                # 计算得分
                P_fan_week = P_fan[active_indices]
                judge_pct = td['judge_score_pct'][active_indices]
                judge_rank = td['judge_rank_score'][active_indices]

                if use_pct:
                    # 百分比法
                    S = judge_pct + P_fan_week
                else:
                    # 排名法
                    diff_mat = P_fan_week[:, None] - P_fan_week[None, :]
                    soft_rank = np.sum(1 / (1 + np.exp(-diff_mat / 0.1)), axis=1) - 0.5
                    R_fan = soft_rank / (n_active - 1) if n_active > 1 else np.array([0.5])
                    S = judge_rank + R_fan

                if use_save and n_active >= 2:
                    # 评委拯救：找出 Bottom 2，评委选择淘汰技术分更低的
                    bottom2_idx = np.argsort(S)[:2]
                    # 比较评委分，淘汰评委分更低的
                    if judge_pct[bottom2_idx[0]] < judge_pct[bottom2_idx[1]]:
                        elim_idx = bottom2_idx[0]
                    else:
                        elim_idx = bottom2_idx[1]
                    elim_this_week = [active_celebs_list[elim_idx]]
                else:
                    # 直接淘汰得分最低的
                    elim_indices = np.argsort(S)[:n_elim]
                    elim_this_week = [active_celebs_list[i] for i in elim_indices]

                for cid in elim_this_week:
                    eliminated.add(cid)
                    elimination_week[cid] = wd['week']

            # 记录结果
            max_week = season_weeks[-1]['week']
            for cid in all_celebs:
                if elimination_week[cid] is not None:
                    results[cid][scenario].append(elimination_week[cid])
                else:
                    # 存活到最后（冠军），记录为最后一周+1
                    results[cid][scenario].append(max_week + 1)

    return results


def compute_survival_probability(sim_results, max_week):
    """
    计算生存概率曲线

    返回:
        dict: {scenario: [prob_week_0, prob_week_1, ...]}
    """
    survival = {}
    for scenario in ['A', 'B', 'C', 'D']:
        elim_weeks = sim_results[scenario]
        n_sims = len(elim_weeks)
        if n_sims == 0:
            survival[scenario] = [1.0] * (max_week + 2)
            continue

        probs = []
        for w in range(max_week + 2):
            # 在第w周存活的概率 = 淘汰周 > w 的比例
            survived = sum(1 for ew in elim_weeks if ew > w)
            probs.append(survived / n_sims)
        survival[scenario] = probs

    return survival


def plot_survival_curves(datas, celeb_id, celeb_name, target_season,
                         actual_season, output_dir, n_simulations=1000):
    """
    绘制生存曲线对比图
    """
    setup_plot_style()

    # 运行模拟
    sim_results = simulate_season_4scenarios(datas, target_season, n_simulations)
    if sim_results is None or celeb_id not in sim_results:
        print(f"  Warning: No simulation results for {celeb_name}")
        return

    celeb_results = sim_results[celeb_id]

    # 获取最大周数
    season_weeks = [wd for wd in datas['train_data']['week_data']
                    if wd['season'] == target_season]
    max_week = max(wd['week'] for wd in season_weeks)

    # 计算生存概率
    survival = compute_survival_probability(celeb_results, max_week)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    weeks = list(range(max_week + 2))
    colors = {'A': '#3498DB', 'B': '#E74C3C', 'C': '#2ECC71', 'D': '#9B59B6'}
    labels = {
        'A': 'Rank Only',
        'B': 'Percent Only',
        'C': 'Rank + Judge Save',
        'D': 'Percent + Judge Save'
    }
    linestyles = {'A': '-', 'B': '-', 'C': '--', 'D': '--'}

    for scenario in ['A', 'B', 'C', 'D']:
        ax.plot(weeks, survival[scenario], color=colors[scenario],
                linestyle=linestyles[scenario], linewidth=2.5, marker='o',
                markersize=6, label=labels[scenario])

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Survival Probability', fontsize=12)
    ax.set_title(f'Survival Curves Comparison: {celeb_name} (S{actual_season})\n'
                 f'Based on {n_simulations} MCMC Simulations', fontsize=13)
    ax.set_xlim(-0.5, max_week + 1.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(weeks)
    ax.set_xticklabels([f'W{w}' for w in weeks])
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = []
    for scenario in ['A', 'B', 'C', 'D']:
        elim_weeks = celeb_results[scenario]
        if elim_weeks:
            mean_elim = np.mean(elim_weeks)
            stats_text.append(f"{labels[scenario]}: E[elim]={mean_elim:.1f}")

    textstr = '\n'.join(stats_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    filename = f'fig_survival_{celeb_name.replace(" ", "_")}_S{actual_season}.png'
    plt.savefig(f'{output_dir}/{filename}', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/{filename}")

    return celeb_results


def plot_scenario_heatmap(all_results, celeb_info, output_dir):
    """
    绘制名次偏移热力图 (4场景 × 4选手)

    参数:
        all_results: dict {celeb_id: {scenario: [elim_weeks]}}
        celeb_info: list of (celeb_id, name, actual_season)
    """
    setup_plot_style()

    scenarios = ['A', 'B', 'C', 'D']
    scenario_labels = ['Rank\nOnly', 'Percent\nOnly', 'Rank+\nSave', 'Percent+\nSave']

    n_celebs = len(celeb_info)
    n_scenarios = len(scenarios)

    # 创建数据矩阵（期望淘汰周）
    data = np.zeros((n_celebs, n_scenarios))
    celeb_labels = []

    for i, (cid, name, actual_season) in enumerate(celeb_info):
        celeb_labels.append(f'{name}\n(S{actual_season})')
        if cid in all_results:
            for j, scenario in enumerate(scenarios):
                elim_weeks = all_results[cid][scenario]
                if elim_weeks:
                    data[i, j] = np.mean(elim_weeks)
                else:
                    data[i, j] = np.nan

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用反转的颜色映射（存活越久颜色越亮）
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto')

    # 设置刻度
    ax.set_xticks(np.arange(n_scenarios))
    ax.set_yticks(np.arange(n_celebs))
    ax.set_xticklabels(scenario_labels, fontsize=11)
    ax.set_yticklabels(celeb_labels, fontsize=10)

    # 添加数值标注
    for i in range(n_celebs):
        for j in range(n_scenarios):
            if not np.isnan(data[i, j]):
                text = ax.text(j, i, f'{data[i, j]:.1f}',
                               ha='center', va='center', fontsize=12,
                               fontweight='bold', color='black')

    ax.set_title('Expected Elimination Week by Scenario\n'
                 '(Higher = Survived Longer, Green = Better)', fontsize=13)

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Expected Elimination Week', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_scenario_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_scenario_heatmap.png")
