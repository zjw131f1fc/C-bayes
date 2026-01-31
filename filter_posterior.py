"""后验筛选：只保留能复现淘汰结果的样本，输出P_fan分布"""

import numpy as np
from src.utils import load_config, load_data, save_data

# 争议周阈值：复现率低于此值视为争议周
CONTROVERSIAL_THRESHOLD = 0.01  # 1%


def filter_posterior_samples(results):
    """筛选能复现淘汰结果的后验样本"""
    S_samples = results['S_samples']  # [n_samples, n_obs]
    P_fan_samples = results['P_fan_samples']  # [n_samples, n_obs]
    train_data = results['train_data']
    week_data = train_data['week_data']

    n_samples, n_obs = S_samples.shape

    # 记录每周的有效样本索引
    week_valid_samples = []  # list of arrays
    week_info = []

    for w, wd in enumerate(week_data):
        if wd['n_eliminated'] == 0:
            week_valid_samples.append(np.arange(n_samples))  # 无淘汰周，所有样本有效
            week_info.append({
                'season': wd.get('season', -1),
                'week': wd.get('week', -1),
                'n_valid': n_samples,
                'rate': 1.0,
                'controversial': False
            })
            continue

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']
        judge_save = wd.get('judge_save_active', False)

        week_obs = np.where(obs_mask)[0]
        true_elim = set(np.where(elim_mask)[0])

        valid_samples = []
        for s in range(n_samples):
            S_week = S_samples[s, obs_mask]

            if judge_save:
                # 评委拯救：预测危险区（前2名）
                pred_danger_local = np.argsort(S_week)[:2]
                pred_danger_global = set(week_obs[pred_danger_local])
                # 真实淘汰者必须在危险区内
                if true_elim.issubset(pred_danger_global):
                    valid_samples.append(s)
            else:
                # 正常淘汰：预测淘汰者
                pred_elim_local = np.argsort(S_week)[:n_elim]
                pred_elim_global = set(week_obs[pred_elim_local])
                if pred_elim_global == true_elim:
                    valid_samples.append(s)

        valid_samples = np.array(valid_samples)
        week_valid_samples.append(valid_samples)

        rate = len(valid_samples) / n_samples
        week_info.append({
            'season': wd.get('season', -1),
            'week': wd.get('week', -1),
            'n_valid': len(valid_samples),
            'rate': rate,
            'controversial': rate < CONTROVERSIAL_THRESHOLD  # 复现率低于阈值视为争议周
        })

    return week_valid_samples, week_info


def compute_filtered_pfan(results, week_valid_samples):
    """计算筛选后的P_fan分布"""
    P_fan_samples = results['P_fan_samples']
    train_data = results['train_data']
    week_data = train_data['week_data']

    n_samples, n_obs = P_fan_samples.shape

    # 输出结构
    pfan_output = {
        'mean': np.zeros(n_obs, dtype=np.float32),
        'std': np.zeros(n_obs, dtype=np.float32),
        'ci_lower': np.zeros(n_obs, dtype=np.float32),
        'ci_upper': np.zeros(n_obs, dtype=np.float32),
        'n_valid_samples': np.zeros(n_obs, dtype=np.int32),
    }

    for w, wd in enumerate(week_data):
        obs_mask = wd['obs_mask']
        week_obs = np.where(obs_mask)[0]
        valid_samples = week_valid_samples[w]

        if len(valid_samples) == 0:
            # 争议性周：使用全部样本（标记为不确定）
            valid_samples = np.arange(n_samples)

        # 筛选后的P_fan样本
        P_fan_filtered = P_fan_samples[valid_samples][:, week_obs]

        # 计算统计量
        pfan_output['mean'][week_obs] = P_fan_filtered.mean(axis=0)
        pfan_output['std'][week_obs] = P_fan_filtered.std(axis=0)
        pfan_output['ci_lower'][week_obs] = np.percentile(P_fan_filtered, 2.5, axis=0)
        pfan_output['ci_upper'][week_obs] = np.percentile(P_fan_filtered, 97.5, axis=0)
        pfan_output['n_valid_samples'][week_obs] = len(valid_samples)

    return pfan_output


def compute_filtered_s(results, week_valid_samples):
    """计算筛选后的S分布（综合得分）"""
    S_samples = results['S_samples']
    train_data = results['train_data']
    week_data = train_data['week_data']

    n_samples, n_obs = S_samples.shape

    # 输出结构
    s_output = {
        'mean': np.zeros(n_obs, dtype=np.float32),
        'std': np.zeros(n_obs, dtype=np.float32),
        'ci_lower': np.zeros(n_obs, dtype=np.float32),
        'ci_upper': np.zeros(n_obs, dtype=np.float32),
        'n_valid_samples': np.zeros(n_obs, dtype=np.int32),
    }

    for w, wd in enumerate(week_data):
        obs_mask = wd['obs_mask']
        week_obs = np.where(obs_mask)[0]
        valid_samples = week_valid_samples[w]

        if len(valid_samples) == 0:
            # 争议性周：使用全部样本
            valid_samples = np.arange(n_samples)

        # 筛选后的S样本
        S_filtered = S_samples[valid_samples][:, week_obs]

        # 计算统计量
        s_output['mean'][week_obs] = S_filtered.mean(axis=0)
        s_output['std'][week_obs] = S_filtered.std(axis=0)
        s_output['ci_lower'][week_obs] = np.percentile(S_filtered, 2.5, axis=0)
        s_output['ci_upper'][week_obs] = np.percentile(S_filtered, 97.5, axis=0)
        s_output['n_valid_samples'][week_obs] = len(valid_samples)

    return s_output


def main():
    print("=" * 60)
    print("后验筛选：生成P_fan分布")
    print("=" * 60)

    # 加载结果
    results = load_data('outputs/results/results.pkl')

    # 筛选后验样本
    print("\n1. 筛选后验样本...")
    week_valid_samples, week_info = filter_posterior_samples(results)

    # 统计
    n_weeks = len(week_info)
    n_controversial = sum(1 for info in week_info if info['controversial'])
    avg_rate = np.mean([info['rate'] for info in week_info])

    print(f"   总周数: {n_weeks}")
    print(f"   争议性周（复现率<{CONTROVERSIAL_THRESHOLD*100:.0f}%）: {n_controversial}")
    print(f"   平均复现率: {avg_rate*100:.1f}%")

    # 计算筛选后的P_fan分布
    print("\n2. 计算筛选后P_fan分布...")
    pfan_output = compute_filtered_pfan(results, week_valid_samples)

    # 计算筛选后的S分布
    print("\n3. 计算筛选后S分布...")
    s_output = compute_filtered_s(results, week_valid_samples)

    # 验证一致性（用筛选后样本的均值）
    print("\n3. 验证筛选后一致性...")
    train_data = results['train_data']
    week_data = train_data['week_data']
    S_samples = results['S_samples']

    correct = 0
    total = 0
    for w, wd in enumerate(week_data):
        if wd['n_eliminated'] == 0:
            continue
        if week_info[w]['controversial']:
            continue  # 跳过争议性周

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']

        week_obs = np.where(obs_mask)[0]
        true_elim = set(np.where(elim_mask)[0])
        valid_samples = week_valid_samples[w]

        # 用筛选后样本的S均值
        S_filtered_mean = S_samples[valid_samples][:, week_obs].mean(axis=0)
        pred_elim_local = np.argsort(S_filtered_mean)[:n_elim]
        pred_elim_global = set(week_obs[pred_elim_local])

        total += 1
        if pred_elim_global == true_elim:
            correct += 1

    print(f"   非争议性周一致性: {correct}/{total} ({correct/total*100:.1f}%)")

    # 保存结果
    print("\n5. 保存结果...")
    results['pfan_filtered'] = pfan_output
    results['s_filtered'] = s_output
    results['week_filter_info'] = week_info
    save_data(results, 'outputs/results/results.pkl')

    # 输出争议性周列表
    print("\n" + "-" * 60)
    print(f"争议性周列表（复现率<{CONTROVERSIAL_THRESHOLD*100:.0f}%）:")
    print("-" * 60)
    for info in week_info:
        if info['controversial']:
            print(f"  Season {info['season']}, Week {info['week']}, rate={info['rate']*100:.2f}%")

    # 输出P_fan统计
    print("\n" + "-" * 60)
    print("P_fan分布统计:")
    print("-" * 60)
    print(f"  均值范围: [{pfan_output['mean'].min():.4f}, {pfan_output['mean'].max():.4f}]")
    print(f"  平均CI宽度: {(pfan_output['ci_upper'] - pfan_output['ci_lower']).mean():.4f}")

    # 不确定性分析
    print("\n" + "-" * 60)
    print("不确定性分析:")
    print("-" * 60)

    ci_width = pfan_output['ci_upper'] - pfan_output['ci_lower']

    # 按淘汰状态分析
    eliminated = np.zeros(len(ci_width), dtype=bool)
    for wd in week_data:
        eliminated[wd['eliminated_mask']] = True

    ci_elim = ci_width[eliminated]
    ci_surv = ci_width[~eliminated]

    print(f"  淘汰者CI宽度: {ci_elim.mean():.4f} ± {ci_elim.std():.4f}")
    print(f"  幸存者CI宽度: {ci_surv.mean():.4f} ± {ci_surv.std():.4f}")

    # 按赛制分析
    ci_pct = []  # 百分比法
    ci_rank = []  # 排名法
    for w, wd in enumerate(week_data):
        obs_mask = wd['obs_mask']
        if wd['rule_method'] == 1:
            ci_pct.extend(ci_width[obs_mask])
        else:
            ci_rank.extend(ci_width[obs_mask])

    print(f"  百分比法CI宽度: {np.mean(ci_pct):.4f}")
    print(f"  排名法CI宽度: {np.mean(ci_rank):.4f}")

    # 按有效样本数分析
    low_samples = pfan_output['n_valid_samples'] < 1000
    high_samples = pfan_output['n_valid_samples'] >= 1000

    print(f"  低样本数(<1000)CI宽度: {ci_width[low_samples].mean():.4f}")
    print(f"  高样本数(>=1000)CI宽度: {ci_width[high_samples].mean():.4f}")

    print("\n" + "=" * 60)
    print("完成！结果已保存到 outputs/results/results.pkl")
    print("=" * 60)


if __name__ == '__main__':
    main()
