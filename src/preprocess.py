"""数据预处理"""

import numpy as np
import pickle


def load_data(path, datas):
    """从文件加载数据"""
    with open(path, 'rb') as f:
        loaded = pickle.load(f)
    datas['train_data'] = loaded['train_data']
    td = datas['train_data']
    print(f"Loaded data: {td['n_obs']} obs, {td['n_weeks']} weeks, "
          f"{td['n_celebs']} celebs, {td['n_pros']} pros, {td['n_seasons']} seasons")
    return datas


def validate_data(datas):
    """校验数据是否符合接口规范"""
    td = datas['train_data']
    errors = []

    # 1. 维度一致性
    n_obs = td['n_obs']
    n_celebs = td['n_celebs']
    n_pros = td['n_pros']
    n_weeks = td['n_weeks']

    if len(td['celeb_idx']) != n_obs:
        errors.append(f"celeb_idx 长度 {len(td['celeb_idx'])} != n_obs {n_obs}")
    if len(td['pro_idx']) != n_obs:
        errors.append(f"pro_idx 长度 {len(td['pro_idx'])} != n_obs {n_obs}")
    if len(td['week_idx']) != n_obs:
        errors.append(f"week_idx 长度 {len(td['week_idx'])} != n_obs {n_obs}")
    if len(td['season_idx']) != n_obs:
        errors.append(f"season_idx 长度 {len(td['season_idx'])} != n_obs {n_obs}")
    if td['X_celeb'].shape[0] != n_celebs:
        errors.append(f"X_celeb 第一维 {td['X_celeb'].shape[0]} != n_celebs {n_celebs}")
    if td['X_pro'].shape[0] != n_pros:
        errors.append(f"X_pro 第一维 {td['X_pro'].shape[0]} != n_pros {n_pros}")
    if td['X_obs'].shape[0] != n_obs:
        errors.append(f"X_obs 第一维 {td['X_obs'].shape[0]} != n_obs {n_obs}")
    if len(td['week_data']) != n_weeks:
        errors.append(f"week_data 长度 {len(td['week_data'])} != n_weeks {n_weeks}")

    # 2. 索引范围
    if td['celeb_idx'].min() < 0 or td['celeb_idx'].max() >= n_celebs:
        errors.append(f"celeb_idx 范围 [{td['celeb_idx'].min()}, {td['celeb_idx'].max()}] 超出 [0, {n_celebs})")
    if td['pro_idx'].min() < 0 or td['pro_idx'].max() >= n_pros:
        errors.append(f"pro_idx 范围 [{td['pro_idx'].min()}, {td['pro_idx'].max()}] 超出 [0, {n_pros})")
    if td['season_idx'].min() < 0 or td['season_idx'].max() >= td['n_seasons']:
        errors.append(f"season_idx 范围超出 [0, {td['n_seasons']})")

    # 3. 数据类型
    if td['celeb_idx'].dtype != np.int32:
        errors.append(f"celeb_idx 类型 {td['celeb_idx'].dtype} != int32")
    if td['pro_idx'].dtype != np.int32:
        errors.append(f"pro_idx 类型 {td['pro_idx'].dtype} != int32")
    if td['X_celeb'].dtype != np.float32:
        errors.append(f"X_celeb 类型 {td['X_celeb'].dtype} != float32")
    if td['X_pro'].dtype != np.float32:
        errors.append(f"X_pro 类型 {td['X_pro'].dtype} != float32")
    if td['X_obs'].dtype != np.float32:
        errors.append(f"X_obs 类型 {td['X_obs'].dtype} != float32")

    # 4. 特征名称数量匹配
    if len(td['X_celeb_names']) != td['X_celeb'].shape[1]:
        errors.append(f"X_celeb_names 数量 {len(td['X_celeb_names'])} != X_celeb 列数 {td['X_celeb'].shape[1]}")
    if len(td['X_pro_names']) != td['X_pro'].shape[1]:
        errors.append(f"X_pro_names 数量 {len(td['X_pro_names'])} != X_pro 列数 {td['X_pro'].shape[1]}")
    if len(td['X_obs_names']) != td['X_obs'].shape[1]:
        errors.append(f"X_obs_names 数量 {len(td['X_obs_names'])} != X_obs 列数 {td['X_obs'].shape[1]}")

    # 5. 淘汰数据一致性
    for i, wd in enumerate(td['week_data']):
        elim_count = wd['eliminated_mask'].sum()
        if elim_count != wd['n_eliminated']:
            errors.append(f"week_data[{i}] eliminated_mask.sum()={elim_count} != n_eliminated={wd['n_eliminated']}")
        if wd['obs_mask'].sum() != wd['n_contestants']:
            errors.append(f"week_data[{i}] obs_mask.sum() != n_contestants")

    # 6. 标准化检查（警告，不报错）
    warnings = []
    for i, name in enumerate(td['X_obs_names']):
        col = td['X_obs'][:, i]
        if abs(col.mean()) > 0.5:
            warnings.append(f"X_obs[{name}] 均值 {col.mean():.2f} 偏离 0")
        if abs(col.std() - 1) > 0.5:
            warnings.append(f"X_obs[{name}] 标准差 {col.std():.2f} 偏离 1")

    # 输出结果
    if errors:
        print(f"数据校验失败，发现 {len(errors)} 个错误:")
        for e in errors:
            print(f"  - {e}")
        raise ValueError("数据校验失败")
    else:
        print(f"数据校验通过")
        if warnings:
            print(f"警告 ({len(warnings)} 个):")
            for w in warnings:
                print(f"  - {w}")

    return datas


def load_mock_data(config, datas):
    """生成符合数据接口规范的模拟数据"""
    np.random.seed(42)

    # 维度
    n_celebs = 36
    n_pros = 15
    n_seasons = 3
    weeks_per_season = 10

    # 生成观测（模拟3赛季，每赛季12人10周，逐周淘汰）
    obs_celeb, obs_pro, obs_week, obs_season = [], [], [], []
    elim_flags = []

    for season in range(n_seasons):
        active = list(range(season * 12, (season + 1) * 12))
        celeb_to_pro = {c: np.random.randint(0, n_pros) for c in active}

        for week in range(weeks_per_season):
            for c in active:
                obs_celeb.append(c)
                obs_pro.append(celeb_to_pro[c])
                obs_week.append(week)  # 赛季内的周数
                obs_season.append(season)
                elim_flags.append(False)

            # 淘汰1人
            if week < 9 and len(active) > 3:
                elim_idx = len(elim_flags) - len(active) + np.random.randint(len(active))
                elim_flags[elim_idx] = True
                active.remove(obs_celeb[elim_idx])

    n_obs = len(obs_celeb)
    celeb_idx = np.array(obs_celeb, dtype=np.int32)
    pro_idx = np.array(obs_pro, dtype=np.int32)
    week_idx = np.array(obs_week, dtype=np.int32)
    season_idx = np.array(obs_season, dtype=np.int32)
    elim_flags = np.array(elim_flags, dtype=bool)

    # 特征矩阵（已标准化）
    X_celeb = np.random.randn(n_celebs, 5).astype(np.float32)
    X_celeb_names = ['is_male', 'age_centered', 'industry_actor', 'industry_athlete', 'log_us_state_pop']

    X_pro = np.random.randn(n_pros, 2).astype(np.float32)
    X_pro_names = ['pro_prev_wins', 'pro_avg_rank']

    X_obs = np.random.randn(n_obs, 4).astype(np.float32)
    X_obs_names = ['z_score', 'score_trend', 'weeks_survived', 'teflon_factor']

    # 评委分数据
    judge_score_pct = np.random.uniform(0.05, 0.15, n_obs).astype(np.float32)
    judge_rank_score = np.random.uniform(0, 1, n_obs).astype(np.float32)

    # 周级数据（按赛季+周组合）
    week_data = []
    for s in range(n_seasons):
        for w in range(weeks_per_season):
            mask = (season_idx == s) & (week_idx == w)
            week_data.append({
                'obs_mask': mask,
                'n_contestants': int(mask.sum()),
                'n_eliminated': int((elim_flags & mask).sum()),
                'eliminated_mask': elim_flags & mask,
                'rule_method': 1 if s < 2 else 0,  # 前2赛季百分比法，第3赛季排名法
                'judge_save_active': s >= 2,
                'season': s,
                'week': w,
            })

    n_weeks = len(week_data)

    datas['train_data'] = {
        'n_obs': n_obs,
        'n_weeks': n_weeks,
        'n_celebs': n_celebs,
        'n_pros': n_pros,
        'n_seasons': n_seasons,
        'celeb_idx': celeb_idx,
        'pro_idx': pro_idx,
        'week_idx': week_idx,
        'season_idx': season_idx,
        'X_celeb': X_celeb,
        'X_pro': X_pro,
        'X_obs': X_obs,
        'X_celeb_names': X_celeb_names,
        'X_pro_names': X_pro_names,
        'X_obs_names': X_obs_names,
        'judge_score_pct': judge_score_pct,
        'judge_rank_score': judge_rank_score,
        'week_data': week_data,
    }

    print(f"Mock data: {n_obs} obs, {n_weeks} weeks, {n_celebs} celebs, {n_pros} pros, {n_seasons} seasons")
    return datas


def filter_data(datas, test_season):
    """
    按赛季划分数据
    输入: test_season - 留出作为测试集的赛季索引
    返回: (train_datas, test_datas)
    """
    td = datas['train_data']
    season_idx = td['season_idx']

    train_mask = (season_idx != test_season)
    test_mask = (season_idx == test_season)

    train_datas = _filter_by_mask(td, train_mask)
    test_datas = _filter_by_mask(td, test_mask)

    return train_datas, test_datas


def _filter_by_mask(td, mask):
    """根据观测级 mask 过滤数据"""
    # 过滤观测级数据
    new_td = {
        'n_obs': int(mask.sum()),
        'n_celebs': td['n_celebs'],
        'n_pros': td['n_pros'],
        'celeb_idx': td['celeb_idx'][mask],
        'pro_idx': td['pro_idx'][mask],
        'week_idx': td['week_idx'][mask],
        'season_idx': td['season_idx'][mask],
        'X_celeb': td['X_celeb'],  # 保留全部（名人可能跨赛季）
        'X_pro': td['X_pro'],      # 保留全部
        'X_obs': td['X_obs'][mask],
        'X_celeb_names': td['X_celeb_names'],
        'X_pro_names': td['X_pro_names'],
        'X_obs_names': td['X_obs_names'],
        'judge_score_pct': td['judge_score_pct'][mask],
        'judge_rank_score': td['judge_rank_score'][mask],
    }

    # 过滤周级数据
    new_week_data = []
    for wd in td['week_data']:
        # 该周是否有观测在 mask 中
        week_mask = wd['obs_mask'] & mask
        if week_mask.sum() == 0:
            continue
        new_week_data.append({
            'obs_mask': week_mask[mask],  # 相对于新的 n_obs
            'n_contestants': int(week_mask.sum()),
            'n_eliminated': int((wd['eliminated_mask'] & mask).sum()),
            'eliminated_mask': (wd['eliminated_mask'] & mask)[mask],
            'rule_method': wd['rule_method'],
            'judge_save_active': wd['judge_save_active'],
            'season': wd['season'],
            'week': wd['week'],
        })

    new_td['n_weeks'] = len(new_week_data)
    new_td['n_seasons'] = len(set(new_td['season_idx']))
    new_td['week_data'] = new_week_data

    return {'train_data': new_td}
