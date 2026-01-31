"""
第三问可视化函数
分析职业舞者与名人特征对评委评分和粉丝投票的影响
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def analyze_feature_significance(datas, output_dir):
    """
    分析特征显著性：比较贝叶斯模型（预测P_fan）和线性模型（预测评委分）的系数

    贝叶斯模型：特征 -> P_fan（粉丝投票比例）
    线性模型：特征 -> judge_score（评委分）

    通过比较两个模型的系数，可以发现：
    - 哪些特征对粉丝投票影响大，但对评委分影响小（粉丝偏好）
    - 哪些特征对评委分影响大，但对粉丝投票影响小（评委偏好）
    - 哪些特征对两者影响一致

    注意：
    - 贝叶斯模型系数 beta_obs 从 posterior_samples 获取（模型参数，不需要筛选）
    - P_fan 数据优先使用筛选后的 pfan_filtered['mean']
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 提取贝叶斯模型的特征系数（模型参数，不需要筛选）
    posterior = datas['posterior_samples']
    beta_obs = posterior['beta_obs']  # [n_samples, n_features]
    bayes_feature_names = datas['X_obs_names_filtered']

    # 获取 P_fan 数据：优先使用筛选后的数据
    if 'pfan_filtered' in datas and 'mean' in datas['pfan_filtered']:
        P_fan_mean = datas['pfan_filtered']['mean']
        P_fan_std = datas['pfan_filtered']['std']
        P_fan_ci_lower = datas['pfan_filtered']['ci_lower']
        P_fan_ci_upper = datas['pfan_filtered']['ci_upper']
        pfan_source = "pfan_filtered (筛选后)"
    elif 'P_fan_samples' in datas:
        P_fan_samples = datas['P_fan_samples']
        P_fan_mean = np.mean(P_fan_samples, axis=0)
        P_fan_std = np.std(P_fan_samples, axis=0)
        P_fan_ci_lower = np.percentile(P_fan_samples, 2.5, axis=0)
        P_fan_ci_upper = np.percentile(P_fan_samples, 97.5, axis=0)
        pfan_source = "P_fan_samples (原始)"
        print("  Warning: pfan_filtered not found, using raw P_fan_samples")
    else:
        P_fan_mean = None
        P_fan_std = None
        P_fan_ci_lower = None
        P_fan_ci_upper = None
        pfan_source = "N/A"
        print("  Warning: No P_fan data found")

    # 贝叶斯模型系数统计（所有特征，包括交互项）
    beta_mean_all = np.mean(beta_obs, axis=0)
    beta_std_all = np.std(beta_obs, axis=0)
    beta_lower_all = np.percentile(beta_obs, 2.5, axis=0)
    beta_upper_all = np.percentile(beta_obs, 97.5, axis=0)
    bayes_significant_all = (beta_lower_all > 0) | (beta_upper_all < 0)

    # 2. 训练线性模型预测评委分
    td = datas['train_data']
    X_obs = td['X_obs']  # [n_obs, n_features]
    linear_feature_names = td['X_obs_names']
    judge_score = td['judge_score_pct']  # [n_obs]

    # 标准化特征（与贝叶斯模型一致）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_obs)

    # Ridge回归（带正则化，避免过拟合）
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, judge_score)

    # 线性模型系数
    linear_coef_all = ridge.coef_

    # 3. 找到两个模型共有的特征，计算系数差异
    # 贝叶斯模型有交互项，线性模型只有基础特征
    # 只比较共有的基础特征
    common_features = []
    bayes_idx = []
    linear_idx = []
    for i, name in enumerate(linear_feature_names):
        if name in bayes_feature_names:
            common_features.append(name)
            bayes_idx.append(bayes_feature_names.index(name))
            linear_idx.append(i)

    # 提取共有特征的系数
    beta_mean = beta_mean_all[bayes_idx]
    beta_std = beta_std_all[bayes_idx]
    beta_lower = beta_lower_all[bayes_idx]
    beta_upper = beta_upper_all[bayes_idx]
    bayes_significant = bayes_significant_all[bayes_idx]
    linear_coef = linear_coef_all[linear_idx]
    feature_names = common_features

    # 归一化系数以便比较（除以各自的最大绝对值）
    beta_normalized = beta_mean / (np.abs(beta_mean).max() + 1e-8)
    linear_normalized = linear_coef / (np.abs(linear_coef).max() + 1e-8)

    # 差异 = 贝叶斯系数 - 线性系数（正值表示粉丝更看重，负值表示评委更看重）
    coef_diff = beta_normalized - linear_normalized

    # 特征分类
    PRO_FEATURES = {'same_sex_pair', 'pro_prev_wins', 'pro_avg_rank', 'pro_is_male'}
    CELEB_FEATURES = {'celeb_fans_week_ratio', 'has_youtube_video', 'youtube_view_count_norm', 'youtube_comment_count_norm'}
    SEASON_FEATURES = {'season_era', 'rule_method'}
    # 其余为比赛动态特征

    def get_feature_category(name):
        base_name = name.split('_x_')[0] if '_x_' in name else name
        if base_name in PRO_FEATURES or any(base_name.startswith(p) for p in ['pro_']):
            return '舞者'
        elif base_name in CELEB_FEATURES or any(base_name.startswith(c) for c in ['celeb_', 'youtube_', 'has_youtube']):
            return '名人'
        elif base_name in SEASON_FEATURES:
            return '赛季'
        else:
            return '动态'

    # 4. 输出结果
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("第三问：特征对评委评分 vs 粉丝投票的影响分析")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("数据来源：")
    output_lines.append(f"  - P_fan 数据: {pfan_source}")
    output_lines.append("  - 贝叶斯系数: posterior_samples['beta_obs'] (模型参数，不需要筛选)")
    output_lines.append(f"  - 贝叶斯模型特征数: {len(bayes_feature_names)} (包含交互项)")
    output_lines.append(f"  - 线性模型特征数: {len(linear_feature_names)} (基础特征)")
    output_lines.append(f"  - 共有特征数: {len(common_features)} (用于比较)")
    output_lines.append("")
    output_lines.append("模型说明：")
    output_lines.append("  - 贝叶斯模型：特征 -> P_fan（粉丝投票比例），系数表示特征对粉丝投票的影响")
    output_lines.append("  - 线性模型：特征 -> judge_score（评委分），系数表示特征对评委评分的影响")
    output_lines.append("  - 差异 = 归一化(贝叶斯系数) - 归一化(线性系数)")
    output_lines.append("    正值：粉丝更看重该特征")
    output_lines.append("    负值：评委更看重该特征")
    output_lines.append("")

    # 4.1 按类别分组展示共有特征
    output_lines.append("-" * 100)
    output_lines.append("【共有特征比较 - 按类别分组】")
    output_lines.append("-" * 100)
    output_lines.append(f"{'类别':<6} {'特征名':<40} {'贝叶斯系数':<12} {'95% CI':<20} {'线性系数':<10} {'差异':<8} {'显著':<4}")
    output_lines.append("-" * 100)

    # 按类别分组
    categories = ['动态', '舞者', '名人', '赛季']
    for cat in categories:
        cat_features = [(i, feature_names[i]) for i in range(len(feature_names)) if get_feature_category(feature_names[i]) == cat]
        if not cat_features:
            continue

        # 按系数绝对值排序
        cat_features.sort(key=lambda x: -abs(beta_mean[x[0]]))

        for idx, name in cat_features:
            b_mean = beta_mean[idx]
            b_ci = f"[{beta_lower[idx]:.3f}, {beta_upper[idx]:.3f}]"
            l_coef = linear_coef[idx]
            diff = coef_diff[idx]
            sig = "Yes" if bayes_significant[idx] else "No"
            output_lines.append(f"{cat:<6} {name:<40} {b_mean:>+.4f}      {b_ci:<20} {l_coef:>+.4f}    {diff:>+.4f}  {sig:<4}")

    output_lines.append("-" * 100)
    output_lines.append("")

    # 4.2 贝叶斯模型交互项（线性模型没有的特征）
    interaction_features = [name for name in bayes_feature_names if name not in linear_feature_names]
    if interaction_features:
        output_lines.append("-" * 100)
        output_lines.append("【贝叶斯模型交互项】（线性模型没有的特征）")
        output_lines.append("-" * 100)
        output_lines.append(f"{'特征名':<55} {'贝叶斯系数':<15} {'95% CI':<25} {'显著':<6}")
        output_lines.append("-" * 100)

        # 按系数绝对值排序
        interaction_idx = [bayes_feature_names.index(name) for name in interaction_features]
        sorted_interaction = sorted(interaction_idx, key=lambda i: -abs(beta_mean_all[i]))

        for idx in sorted_interaction:
            name = bayes_feature_names[idx]
            b_mean = beta_mean_all[idx]
            b_ci = f"[{beta_lower_all[idx]:.3f}, {beta_upper_all[idx]:.3f}]"
            sig = "Yes" if bayes_significant_all[idx] else "No"
            output_lines.append(f"{name:<55} {b_mean:>+.4f}        {b_ci:<25} {sig:<6}")

        output_lines.append("-" * 100)
        output_lines.append("")

    # 5. 分类汇总
    output_lines.append("=" * 100)
    output_lines.append("特征分类汇总")
    output_lines.append("=" * 100)
    output_lines.append("")

    # 粉丝偏好特征（差异 > 0.1 且贝叶斯显著）
    fan_prefer = [(i, feature_names[i], coef_diff[i])
                  for i in range(len(feature_names))
                  if coef_diff[i] > 0.1 and bayes_significant[i]]
    fan_prefer.sort(key=lambda x: -x[2])

    output_lines.append("【粉丝偏好特征】（对粉丝投票影响 > 对评委评分影响）")
    output_lines.append("-" * 60)
    if fan_prefer:
        for idx, name, diff in fan_prefer:
            output_lines.append(f"  {name:<40} 差异: {diff:>+.4f}")
    else:
        output_lines.append("  （无显著特征）")
    output_lines.append("")

    # 评委偏好特征（差异 < -0.1 且贝叶斯显著）
    judge_prefer = [(i, feature_names[i], coef_diff[i])
                    for i in range(len(feature_names))
                    if coef_diff[i] < -0.1 and bayes_significant[i]]
    judge_prefer.sort(key=lambda x: x[2])

    output_lines.append("【评委偏好特征】（对评委评分影响 > 对粉丝投票影响）")
    output_lines.append("-" * 60)
    if judge_prefer:
        for idx, name, diff in judge_prefer:
            output_lines.append(f"  {name:<40} 差异: {diff:>+.4f}")
    else:
        output_lines.append("  （无显著特征）")
    output_lines.append("")

    # 一致性特征（差异绝对值 < 0.1 且贝叶斯显著）
    consistent = [(i, feature_names[i], coef_diff[i])
                  for i in range(len(feature_names))
                  if abs(coef_diff[i]) <= 0.1 and bayes_significant[i]]
    consistent.sort(key=lambda x: -abs(beta_mean[x[0]]))

    output_lines.append("【一致性特征】（对评委和粉丝影响相近）")
    output_lines.append("-" * 60)
    if consistent:
        for idx, name, diff in consistent:
            output_lines.append(f"  {name:<40} 差异: {diff:>+.4f}")
    else:
        output_lines.append("  （无显著特征）")
    output_lines.append("")

    # 6. 名人特征分析（alpha_beta）
    output_lines.append("=" * 100)
    output_lines.append("名人特征分析 (alpha_beta)")
    output_lines.append("=" * 100)
    output_lines.append("")
    output_lines.append("说明：alpha_beta 是名人特征对名人效应 (alpha) 的影响系数")
    output_lines.append("      alpha = X_celeb @ alpha_beta + 随机效应 + 样条效应")
    output_lines.append("")

    # 从 trace 中获取 alpha_beta
    trace = datas.get('trace', {})
    if hasattr(trace, 'posterior') and 'alpha_beta' in trace.posterior:
        alpha_beta = trace.posterior['alpha_beta'].values.reshape(-1, trace.posterior['alpha_beta'].shape[-1])
    elif 'alpha_beta' in trace:
        alpha_beta = trace['alpha_beta']
    else:
        alpha_beta = None

    if alpha_beta is not None:
        celeb_feature_names = td.get('X_celeb_names', [f'celeb_feat_{i}' for i in range(alpha_beta.shape[1])])

        alpha_beta_mean = np.mean(alpha_beta, axis=0)
        alpha_beta_std = np.std(alpha_beta, axis=0)
        alpha_beta_lower = np.percentile(alpha_beta, 2.5, axis=0)
        alpha_beta_upper = np.percentile(alpha_beta, 97.5, axis=0)
        alpha_beta_significant = (alpha_beta_lower > 0) | (alpha_beta_upper < 0)

        output_lines.append("-" * 100)
        output_lines.append(f"{'名人特征名':<55} {'系数':<12} {'95% CI':<25} {'显著':<6}")
        output_lines.append("-" * 100)

        # 按系数绝对值排序
        sorted_idx = np.argsort(-np.abs(alpha_beta_mean))
        for idx in sorted_idx:
            name = celeb_feature_names[idx] if idx < len(celeb_feature_names) else f'celeb_feat_{idx}'
            coef = alpha_beta_mean[idx]
            ci = f"[{alpha_beta_lower[idx]:.3f}, {alpha_beta_upper[idx]:.3f}]"
            sig = "Yes" if alpha_beta_significant[idx] else "No"
            output_lines.append(f"{name:<55} {coef:>+.4f}      {ci:<25} {sig:<6}")

        output_lines.append("-" * 100)
        output_lines.append("")

        # 显著特征汇总
        sig_features = [(celeb_feature_names[i], alpha_beta_mean[i])
                        for i in range(len(alpha_beta_mean)) if alpha_beta_significant[i]]
        if sig_features:
            output_lines.append("【显著名人特征】")
            output_lines.append("-" * 60)
            for name, coef in sorted(sig_features, key=lambda x: -abs(x[1])):
                direction = "正向影响" if coef > 0 else "负向影响"
                output_lines.append(f"  {name:<45} {coef:>+.4f} ({direction})")
            output_lines.append("")
    else:
        output_lines.append("  （alpha_beta 数据不可用）")
        output_lines.append("")

    # 7. 名人/舞者效应分析
    output_lines.append("=" * 100)
    output_lines.append("名人效应 (alpha) 和舞者效应 (delta) 分析")
    output_lines.append("=" * 100)
    output_lines.append("")

    # 名人效应
    alpha = posterior['alpha']  # [n_samples, n_celebs]
    alpha_mean = np.mean(alpha, axis=0)
    alpha_std = np.mean(np.std(alpha, axis=0))

    output_lines.append(f"名人效应 (alpha):")
    output_lines.append(f"  - 名人数量: {alpha.shape[1]}")
    output_lines.append(f"  - 效应均值范围: [{alpha_mean.min():.4f}, {alpha_mean.max():.4f}]")
    output_lines.append(f"  - 平均后验标准差: {alpha_std:.4f}")
    output_lines.append(f"  - 效应方差: {np.var(alpha_mean):.4f}")
    output_lines.append("")

    # 舞者效应
    delta = posterior['delta']  # [n_samples, n_pros]
    delta_mean = np.mean(delta, axis=0)
    delta_std = np.mean(np.std(delta, axis=0))

    output_lines.append(f"舞者效应 (delta):")
    output_lines.append(f"  - 舞者数量: {delta.shape[1]}")
    output_lines.append(f"  - 效应均值范围: [{delta_mean.min():.4f}, {delta_mean.max():.4f}]")
    output_lines.append(f"  - 平均后验标准差: {delta_std:.4f}")
    output_lines.append(f"  - 效应方差: {np.var(delta_mean):.4f}")
    output_lines.append("")

    # 效应比较
    alpha_var = np.var(alpha_mean)
    delta_var = np.var(delta_mean)
    total_var = alpha_var + delta_var

    output_lines.append(f"效应贡献比例:")
    output_lines.append(f"  - 名人效应占比: {100*alpha_var/total_var:.1f}%")
    output_lines.append(f"  - 舞者效应占比: {100*delta_var/total_var:.1f}%")
    output_lines.append("")

    # 7. 年龄样条效应（如果存在）
    if 'celeb_spline_0_coef' in posterior:
        output_lines.append("=" * 100)
        output_lines.append("年龄非线性效应 (样条)")
        output_lines.append("=" * 100)
        output_lines.append("")

        spline_coef = posterior['celeb_spline_0_coef']
        spline_mean = np.mean(spline_coef, axis=0)
        spline_std = np.std(spline_coef, axis=0)

        output_lines.append(f"样条系数数量: {len(spline_mean)}")
        for i, (m, s) in enumerate(zip(spline_mean, spline_std)):
            output_lines.append(f"  - 系数 {i}: {m:>+.4f} ± {s:.4f}")
        output_lines.append("")

    output_lines.append("=" * 100)
    output_lines.append("分析完成")
    output_lines.append("=" * 100)

    # 写入文件
    output_path = f'{output_dir}/feature_analysis.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"  Saved: {output_path}")

    # 打印到控制台
    for line in output_lines:
        print(f"  {line}")

    result = {
        'feature_names': feature_names,
        'beta_mean': beta_mean,
        'beta_std': beta_std,
        'beta_lower': beta_lower,
        'beta_upper': beta_upper,
        'linear_coef': linear_coef,
        'coef_diff': coef_diff,
        'bayes_significant': bayes_significant,
        'pfan_source': pfan_source,
    }

    # 添加 P_fan 数据（如果可用）
    if P_fan_mean is not None:
        result['P_fan_mean'] = P_fan_mean
        result['P_fan_std'] = P_fan_std
        result['P_fan_ci_lower'] = P_fan_ci_lower
        result['P_fan_ci_upper'] = P_fan_ci_upper

    return result
