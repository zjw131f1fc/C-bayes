"""
第一问可视化函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def setup_plot_style():
    """设置绘图样式"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 150


def plot_posterior_trajectory(datas, celeb_name, celeb_id, output_dir):
    """
    图表一：后验估计轨迹图
    展示某位选手整个赛季的粉丝投票百分比估计及不确定性

    参数:
        datas: 包含 P_fan_samples, train_data 的数据字典
        celeb_name: 选手名称（用于标题）
        celeb_id: 选手在 celeb_idx 中的 ID
        output_dir: 输出目录
    """
    setup_plot_style()

    P_fan_samples = datas['P_fan_samples']  # [n_samples, n_obs]
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    week_data = td['week_data']

    # 找到该选手的所有观测点
    celeb_mask = celeb_idx == celeb_id
    celeb_obs_indices = np.where(celeb_mask)[0]

    if len(celeb_obs_indices) == 0:
        print(f"  Warning: No observations found for celeb_id={celeb_id}")
        return

    # 收集每周的数据
    weeks = []
    p_fan_means = []
    p_fan_lower = []
    p_fan_upper = []
    eliminated_week = None

    for obs_idx in celeb_obs_indices:
        # 找到这个观测点属于哪一周
        for w_idx, wd in enumerate(week_data):
            if wd['obs_mask'][obs_idx]:
                week_num = wd['week']
                weeks.append(week_num)

                # 该观测点的 P_fan 后验分布
                p_fan_obs = P_fan_samples[:, obs_idx]
                p_fan_means.append(np.mean(p_fan_obs))

                # 95% HDPI
                lower = np.percentile(p_fan_obs, 2.5)
                upper = np.percentile(p_fan_obs, 97.5)
                p_fan_lower.append(lower)
                p_fan_upper.append(upper)

                # 检查是否被淘汰
                if wd['eliminated_mask'][obs_idx]:
                    eliminated_week = week_num
                break

    # 排序
    sort_idx = np.argsort(weeks)
    weeks = np.array(weeks)[sort_idx]
    p_fan_means = np.array(p_fan_means)[sort_idx]
    p_fan_lower = np.array(p_fan_lower)[sort_idx]
    p_fan_upper = np.array(p_fan_upper)[sort_idx]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 阴影区域（95% CI）
    ax.fill_between(weeks, p_fan_lower, p_fan_upper, alpha=0.3, color='steelblue',
                    label='95% Credible Interval')

    # 中心线（均值）
    ax.plot(weeks, p_fan_means, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Posterior Mean')

    # 淘汰标记
    if eliminated_week is not None:
        elim_idx = np.where(weeks == eliminated_week)[0][0]
        ax.scatter([eliminated_week], [p_fan_means[elim_idx]], marker='X', s=200,
                   color='red', zorder=5, label='Eliminated')

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Estimated Fan Vote Share (P_fan)', fontsize=12)
    ax.set_title(f'Figure 1: Posterior Distribution of Latent Fan Votes for {celeb_name}',
                 fontsize=14)
    ax.legend(loc='best')
    ax.set_xticks(weeks)
    ax.grid(True, alpha=0.3)

    # P_fan 是占比，纵轴固定为 [0, 1]
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_trajectory_{celeb_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig1_trajectory_{celeb_name}.png")


def plot_score_trajectory_with_threshold(datas, celeb_name, celeb_id, output_dir, target_season=None):
    """
    后验得分轨迹图（带淘汰临界线）
    展示某位选手某个赛季的综合得分 S 及其与淘汰临界线的关系

    参数:
        datas: 包含 S_samples, train_data 的数据字典
        celeb_name: 选手名称（用于标题）
        celeb_id: 选手在 celeb_idx 中的 ID
        output_dir: 输出目录
        target_season: 指定赛季（None 则自动选择被淘汰的赛季）
    """
    setup_plot_style()

    S_samples = datas['S_samples']  # [n_samples, n_obs]
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    week_data = td['week_data']
    n_samples = S_samples.shape[0]

    # 找到该选手的所有观测点
    celeb_mask = celeb_idx == celeb_id
    celeb_obs_indices = np.where(celeb_mask)[0]

    if len(celeb_obs_indices) == 0:
        print(f"  Warning: No observations found for celeb_id={celeb_id}")
        return

    # 按赛季分组观测点
    season_obs = {}  # season -> list of (obs_idx, week, eliminated)
    for obs_idx in celeb_obs_indices:
        for wd in week_data:
            if wd['obs_mask'][obs_idx]:
                season = wd['season']
                if season not in season_obs:
                    season_obs[season] = []
                season_obs[season].append({
                    'obs_idx': obs_idx,
                    'week': wd['week'],
                    'eliminated': wd['eliminated_mask'][obs_idx],
                    'week_data': wd,
                })
                break

    # 选择目标赛季
    if target_season is None:
        # 自动选择被淘汰的赛季
        for season, obs_list in season_obs.items():
            if any(o['eliminated'] for o in obs_list):
                target_season = season
                break
        if target_season is None:
            target_season = list(season_obs.keys())[0]

    if target_season not in season_obs:
        print(f"  Warning: Season {target_season} not found for celeb_id={celeb_id}")
        return

    obs_list = season_obs[target_season]

    # 收集该赛季的数据
    weeks = []
    s_means = []
    s_lower = []
    s_upper = []
    threshold_means = []
    threshold_lower = []
    threshold_upper = []
    eliminated_week = None

    for obs_info in obs_list:
        obs_idx = obs_info['obs_idx']
        week_num = obs_info['week']
        wd = obs_info['week_data']
        n_elim = wd['n_eliminated']

        weeks.append(week_num)

        # 该观测点的 S 后验分布
        s_obs = S_samples[:, obs_idx]
        s_means.append(np.mean(s_obs))
        s_lower.append(np.percentile(s_obs, 2.5))
        s_upper.append(np.percentile(s_obs, 97.5))

        # 计算该周的淘汰临界线
        week_mask = wd['obs_mask']
        week_indices = np.where(week_mask)[0]
        n_contestants = len(week_indices)

        if n_elim > 0 and n_contestants > n_elim:
            thresholds = []
            for s in range(n_samples):
                S_week = S_samples[s, week_mask]
                sorted_scores = np.sort(S_week)
                threshold = sorted_scores[n_elim]
                thresholds.append(threshold)

            thresholds = np.array(thresholds)
            threshold_means.append(np.mean(thresholds))
            threshold_lower.append(np.percentile(thresholds, 2.5))
            threshold_upper.append(np.percentile(thresholds, 97.5))
        else:
            threshold_means.append(np.nan)
            threshold_lower.append(np.nan)
            threshold_upper.append(np.nan)

        if obs_info['eliminated']:
            eliminated_week = week_num

    # 排序
    sort_idx = np.argsort(weeks)
    weeks = np.array(weeks)[sort_idx]
    s_means = np.array(s_means)[sort_idx]
    s_lower = np.array(s_lower)[sort_idx]
    s_upper = np.array(s_upper)[sort_idx]
    threshold_means = np.array(threshold_means)[sort_idx]
    threshold_lower = np.array(threshold_lower)[sort_idx]
    threshold_upper = np.array(threshold_upper)[sort_idx]

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 选手得分的阴影区域（95% CI）
    ax.fill_between(weeks, s_lower, s_upper, alpha=0.3, color='steelblue',
                    label='Score 95% CI')

    # 选手得分中心线（均值）
    ax.plot(weeks, s_means, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Score (Posterior Mean)')

    # 淘汰临界线（带置信区间）
    valid_threshold = ~np.isnan(threshold_means)
    if valid_threshold.any():
        ax.fill_between(weeks[valid_threshold],
                        threshold_lower[valid_threshold],
                        threshold_upper[valid_threshold],
                        alpha=0.2, color='red', label='Threshold 95% CI')
        ax.plot(weeks[valid_threshold], threshold_means[valid_threshold],
                's--', color='red', linewidth=1.5, markersize=6,
                label='Elimination Threshold')

    # 淘汰标记
    if eliminated_week is not None:
        elim_idx = np.where(weeks == eliminated_week)[0][0]
        ax.scatter([eliminated_week], [s_means[elim_idx]], marker='X', s=200,
                   color='darkred', zorder=5, label='Eliminated')

    ax.set_xlabel('Week', fontsize=12)
    ax.set_ylabel('Combined Score (S = Judge + Fan)', fontsize=12)
    ax.set_title(f'Posterior Score Trajectory with Elimination Threshold\n{celeb_name} (Season {target_season})',
                 fontsize=13)
    ax.legend(loc='best', fontsize=9)
    ax.set_xticks(weeks)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig_score_trajectory_{celeb_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig_score_trajectory_{celeb_name}.png")


def plot_decision_gap(datas, output_dir):
    """
    图表二：决策缺口散点图
    展示 Acc_A 与 Acc_B 的差异，识别高风险预测点

    横轴: p_i (后验样本中预测正确的比例)
    纵轴: δ_i = |f(S̄) - p_i|
    """
    setup_plot_style()

    week_results = datas['metrics']['week_results']

    # 收集数据
    p_i_list = []
    delta_i_list = []
    correct_list = []  # 点估计是否正确

    for wr in week_results:
        if wr['accuracy'] is None:
            continue

        p_i = wr['accuracy']           # 后验样本准确率
        f_mean = wr['accuracy_mean']   # 点估计准确率 (0 or 1)
        delta_i = wr['decision_gap']   # |f(S̄) - p_i|

        p_i_list.append(p_i)
        delta_i_list.append(delta_i)
        correct_list.append(f_mean == 1.0)

    p_i_arr = np.array(p_i_list)
    delta_i_arr = np.array(delta_i_list)
    correct_arr = np.array(correct_list)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 正确的点（绿色）
    ax.scatter(p_i_arr[correct_arr], delta_i_arr[correct_arr],
               c='green', alpha=0.6, s=50, label='Correct Prediction')

    # 错误的点（红色）
    ax.scatter(p_i_arr[~correct_arr], delta_i_arr[~correct_arr],
               c='red', alpha=0.6, s=50, label='Wrong Prediction')

    # 理论曲线：当 f(S̄)=1 时，δ = 1 - p；当 f(S̄)=0 时，δ = p
    # 最大 δ 在 p=0.5 时达到 0.5
    p_theory = np.linspace(0, 1, 100)
    delta_upper = np.minimum(p_theory, 1 - p_theory)
    ax.plot(p_theory, delta_upper, 'k--', alpha=0.5, label='Theoretical Upper Bound')

    ax.set_xlabel('Predictive Confidence ($p_i$)', fontsize=12)
    ax.set_ylabel('Decision Gap ($\\delta_i = |f(\\bar{S}) - p_i|$)', fontsize=12)
    ax.set_title('Figure 2: Analysis of Local Uncertainty Bias vs. Predictive Confidence',
                 fontsize=14)
    ax.legend(loc='upper center')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.6)
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    n_correct = correct_arr.sum()
    n_total = len(correct_arr)
    textstr = f'Total weeks: {n_total}\nCorrect: {n_correct} ({100*n_correct/n_total:.1f}%)'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_decision_gap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig2_decision_gap.png")


def generate_consistency_table(datas, output_dir):
    """
    图表三：模型一致性总结表
    按赛季类型分组统计 Acc_A, Acc_B, Δ
    """
    week_results = datas['metrics']['week_results']

    # 按赛季类型分组
    groups = {'Early': [], 'Middle': [], 'Late': []}

    for wr in week_results:
        if wr['accuracy'] is None:
            continue
        era = wr['season_era']
        groups[era].append(wr)

    # 计算统计量
    table_data = []
    for era in ['Early', 'Middle', 'Late']:
        results = groups[era]
        if not results:
            continue

        acc_b = np.mean([r['accuracy'] for r in results])
        acc_a = np.mean([r['accuracy_mean'] for r in results])
        delta = acc_a - acc_b
        n_weeks = len(results)

        table_data.append({
            'Season Category': era,
            'Acc_A (Point Est.)': f'{acc_a:.1%}',
            'Acc_B (Posterior)': f'{acc_b:.1%}',
            'Gap (Δ)': f'{delta:+.1%}',
            'N Weeks': n_weeks,
        })

    # 全局统计
    all_results = [r for r in week_results if r['accuracy'] is not None]
    acc_b_all = np.mean([r['accuracy'] for r in all_results])
    acc_a_all = np.mean([r['accuracy_mean'] for r in all_results])
    delta_all = acc_a_all - acc_b_all

    table_data.append({
        'Season Category': 'Overall',
        'Acc_A (Point Est.)': f'{acc_a_all:.1%}',
        'Acc_B (Posterior)': f'{acc_b_all:.1%}',
        'Gap (Δ)': f'{delta_all:+.1%}',
        'N Weeks': len(all_results),
    })

    # 保存为文本
    with open(f'{output_dir}/table_consistency.txt', 'w') as f:
        f.write("Table: Model Consistency Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Season Category':<20} {'Acc_A':<15} {'Acc_B':<15} {'Gap (Δ)':<15} {'N Weeks':<10}\n")
        f.write("-" * 80 + "\n")
        for row in table_data:
            f.write(f"{row['Season Category']:<20} {row['Acc_A (Point Est.)']:<15} "
                    f"{row['Acc_B (Posterior)']:<15} {row['Gap (Δ)']:<15} {row['N Weeks']:<10}\n")
        f.write("=" * 80 + "\n")

    print(f"  Saved: {output_dir}/table_consistency.txt")

    # 同时打印到控制台
    print("\n  Table: Model Consistency Summary")
    print("  " + "=" * 75)
    print(f"  {'Season Category':<18} {'Acc_A':<12} {'Acc_B':<12} {'Gap (Δ)':<12} {'N Weeks':<8}")
    print("  " + "-" * 75)
    for row in table_data:
        print(f"  {row['Season Category']:<18} {row['Acc_A (Point Est.)']:<12} "
              f"{row['Acc_B (Posterior)']:<12} {row['Gap (Δ)']:<12} {row['N Weeks']:<8}")
    print("  " + "=" * 75)

    return table_data


def find_controversial_and_certain_celebs(datas):
    """
    找到争议选手（高不确定性）和高确定性选手

    返回:
        controversial: (celeb_id, celeb_name, avg_variance)
        certain: (celeb_id, celeb_name, avg_variance)
    """
    P_fan_samples = datas['P_fan_samples']
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    n_celebs = td['n_celebs']

    # 计算每个选手的平均 P_fan 方差
    celeb_variances = []
    for c in range(n_celebs):
        mask = celeb_idx == c
        if mask.sum() == 0:
            celeb_variances.append(0)
            continue
        # 该选手所有观测点的 P_fan 方差的平均值
        var_per_obs = np.var(P_fan_samples[:, mask], axis=0)
        celeb_variances.append(np.mean(var_per_obs))

    celeb_variances = np.array(celeb_variances)

    # 找到方差最大和最小的选手（排除方差为0的）
    valid_mask = celeb_variances > 0
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return None, None

    controversial_id = valid_indices[np.argmax(celeb_variances[valid_mask])]
    certain_id = valid_indices[np.argmin(celeb_variances[valid_mask])]

    return (controversial_id, celeb_variances[controversial_id]), \
           (certain_id, celeb_variances[certain_id])


def get_eliminated_celebs(datas, n_celebs=10):
    """
    获取被淘汰的选手列表

    返回:
        list of (celeb_id, eliminated_week, n_weeks_participated)
    """
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    week_data = td['week_data']
    n_total_celebs = td['n_celebs']

    eliminated_celebs = []

    for c in range(n_total_celebs):
        celeb_mask = celeb_idx == c
        celeb_obs_indices = np.where(celeb_mask)[0]

        if len(celeb_obs_indices) == 0:
            continue

        # 检查该选手是否被淘汰
        eliminated_week = None
        weeks_participated = []

        for obs_idx in celeb_obs_indices:
            for wd in week_data:
                if wd['obs_mask'][obs_idx]:
                    weeks_participated.append(wd['week'])
                    if wd['eliminated_mask'][obs_idx]:
                        eliminated_week = wd['week']
                    break

        if eliminated_week is not None and len(weeks_participated) >= 3:
            eliminated_celebs.append((c, eliminated_week, len(weeks_participated)))

    # 按参与周数排序（优先选择参与周数多的）
    eliminated_celebs.sort(key=lambda x: -x[2])

    return eliminated_celebs[:n_celebs]


def get_celebs_by_variance(datas, n_per_group=3):
    """
    按方差分组获取选手：高、中、低方差各 n_per_group 个

    返回:
        list of (celeb_id, variance, group_name)
    """
    P_fan_samples = datas['P_fan_samples']
    td = datas['train_data']
    celeb_idx = td['celeb_idx']
    n_celebs = td['n_celebs']

    # 计算每个选手的平均 P_fan 方差
    celeb_variances = []
    for c in range(n_celebs):
        mask = celeb_idx == c
        if mask.sum() == 0:
            celeb_variances.append(0)
            continue
        var_per_obs = np.var(P_fan_samples[:, mask], axis=0)
        celeb_variances.append(np.mean(var_per_obs))

    celeb_variances = np.array(celeb_variances)

    # 排除方差为0的选手
    valid_mask = celeb_variances > 0
    valid_indices = np.where(valid_mask)[0]
    valid_variances = celeb_variances[valid_mask]

    if len(valid_indices) == 0:
        return []

    # 按方差排序
    sorted_idx = np.argsort(valid_variances)
    n_valid = len(sorted_idx)

    results = []

    # 高方差（最后 n_per_group 个）
    for i in range(min(n_per_group, n_valid)):
        idx = sorted_idx[-(i+1)]
        cid = valid_indices[idx]
        results.append((cid, valid_variances[idx], 'high'))

    # 低方差（最前 n_per_group 个）
    for i in range(min(n_per_group, n_valid)):
        idx = sorted_idx[i]
        cid = valid_indices[idx]
        if (cid, valid_variances[idx], 'high') not in results:
            results.append((cid, valid_variances[idx], 'low'))

    # 中方差（中间 n_per_group 个）
    mid_start = n_valid // 2 - n_per_group // 2
    for i in range(min(n_per_group, n_valid)):
        idx = sorted_idx[mid_start + i]
        if idx < 0 or idx >= n_valid:
            continue
        cid = valid_indices[idx]
        if not any(r[0] == cid for r in results):
            results.append((cid, valid_variances[idx], 'medium'))

    return results


def print_summary_stats(datas):
    """打印第一问需要的汇总统计数据"""
    metrics = datas['metrics']

    print("\n" + "=" * 60)
    print("第一问 - 汇总统计数据")
    print("=" * 60)
    print(f"\n一致性度量指标：")
    print(f"  全局平均点估计准确率 (Acc_A): {metrics['mean_accuracy_expectation']:.1%}")
    print(f"  全局平均后验期望准确率 (Acc_B): {metrics['mean_accuracy']:.1%}")

    print(f"\n不确定性度量：")
    print(f"  全局平均后验样本方差: {metrics['mean_p_fan_var']:.6f}")
    print(f"  全局平均决策风险 (|Acc_A - Acc_B|): {metrics['mean_decision_gap']:.3f}")

    print(f"\n评估周数: {metrics['n_weeks_evaluated']}")
    print("=" * 60)


def plot_posterior_std_bar(datas, output_dir, feature_names=None):
    """
    图表四：后验标准差柱状图
    展示各参数的后验标准差，衡量模型确定性

    参数:
        datas: 包含 posterior_samples 的数据字典
        output_dir: 输出目录
        feature_names: 线性特征名称列表（可选）
    """
    setup_plot_style()

    posterior = datas['posterior_samples']

    # 收集各参数的后验标准差
    param_names = []
    param_stds = []
    param_colors = []

    # 1. 线性特征系数 beta_obs
    if 'beta_obs' in posterior:
        beta_obs = posterior['beta_obs']  # [n_samples, n_features]
        n_features = beta_obs.shape[1]

        # 获取特征名称
        if feature_names is None:
            feature_names = datas.get('X_obs_names_filtered', [f'feat_{i}' for i in range(n_features)])

        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f'feat_{i}'
            std = np.std(beta_obs[:, i])
            param_names.append(name)
            param_stds.append(std)
            param_colors.append('steelblue')

    # 2. 名人效应 alpha（取平均标准差）
    if 'alpha' in posterior:
        alpha = posterior['alpha']  # [n_samples, n_celebs]
        alpha_std_mean = np.mean(np.std(alpha, axis=0))
        param_names.append('α (celeb effect)')
        param_stds.append(alpha_std_mean)
        param_colors.append('coral')

    # 3. 舞者效应 delta（取平均标准差）
    if 'delta' in posterior:
        delta = posterior['delta']  # [n_samples, n_pros]
        delta_std_mean = np.mean(np.std(delta, axis=0))
        param_names.append('δ (pro effect)')
        param_stds.append(delta_std_mean)
        param_colors.append('coral')

    # 4. 温度参数 tau
    if 'tau' in posterior:
        tau = posterior['tau']
        param_names.append('τ (temperature)')
        param_stds.append(np.std(tau))
        param_colors.append('green')

    # 5. 评委拯救参数 theta_save
    if 'theta_save' in posterior:
        theta_save = posterior['theta_save']
        param_names.append('θ_save (judge save)')
        param_stds.append(np.std(theta_save))
        param_colors.append('green')

    # 6. Horseshoe 参数
    if 'tau_hs' in posterior:
        param_names.append('τ_hs (global shrink)')
        param_stds.append(np.std(posterior['tau_hs']))
        param_colors.append('purple')

    if 'c2_hs' in posterior:
        param_names.append('c²_hs (slab var)')
        param_stds.append(np.std(posterior['c2_hs']))
        param_colors.append('purple')

    # 按标准差排序
    sorted_idx = np.argsort(param_stds)[::-1]  # 降序
    param_names = [param_names[i] for i in sorted_idx]
    param_stds = [param_stds[i] for i in sorted_idx]
    param_colors = [param_colors[i] for i in sorted_idx]

    # 绘图
    fig, ax = plt.subplots(figsize=(12, max(6, len(param_names) * 0.3)))

    y_pos = np.arange(len(param_names))
    bars = ax.barh(y_pos, param_stds, color=param_colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=9)
    ax.invert_yaxis()  # 最大的在上面
    ax.set_xlabel('Posterior Standard Deviation', fontsize=12)
    ax.set_title('Figure 4: Parameter Uncertainty (Posterior Std)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')

    # 添加数值标签
    for i, (bar, std) in enumerate(zip(bars, param_stds)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{std:.3f}', va='center', fontsize=8)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Linear Features (β)'),
        Patch(facecolor='coral', alpha=0.7, label='Random Effects (α, δ)'),
        Patch(facecolor='green', alpha=0.7, label='Likelihood Params (τ, θ)'),
        Patch(facecolor='purple', alpha=0.7, label='Horseshoe Params'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_posterior_std.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/fig4_posterior_std.png")

    # 同时输出文本表格
    with open(f'{output_dir}/table_posterior_std.txt', 'w') as f:
        f.write("Table: Parameter Posterior Standard Deviations\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Parameter':<35} {'Std':<15}\n")
        f.write("-" * 60 + "\n")
        for name, std in zip(param_names, param_stds):
            f.write(f"{name:<35} {std:<15.4f}\n")
        f.write("=" * 60 + "\n")
    print(f"  Saved: {output_dir}/table_posterior_std.txt")
