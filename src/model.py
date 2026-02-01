"""贝叶斯模型 (NumPyro 版本) - 向量化实现"""

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from patsy import dmatrix


def _build_spline_basis(x, n_knots, degree):
    """构建B样条基矩阵"""
    knots = np.linspace(x.min(), x.max(), n_knots)
    basis = dmatrix(
        f"bs(x, knots={list(knots[1:-1])}, degree={degree}, include_intercept=False) - 1",
        {"x": x}, return_type='dataframe'
    )
    return np.asarray(basis, dtype=np.float32)


def _build_vectorized_week_data(week_data, n_obs, judge_score_pct, judge_rank_score):
    """预处理week_data为向量化格式"""
    n_weeks = len(week_data)
    max_contestants = max(wd['n_contestants'] for wd in week_data)
    max_elim = max(wd['n_eliminated'] for wd in week_data if wd['n_eliminated'] > 0)

    # 索引数组: [n_weeks, max_contestants]
    week_indices = np.full((n_weeks, max_contestants), -1, dtype=np.int32)
    week_mask = np.zeros((n_weeks, max_contestants), dtype=bool)

    # 淘汰索引: [n_weeks, max_elim] (局部索引)
    elim_local = np.full((n_weeks, max_elim), 0, dtype=np.int32)  # 用0而非-1，避免索引问题
    elim_mask = np.zeros((n_weeks, max_elim), dtype=bool)
    n_elim_per_week = np.zeros(n_weeks, dtype=np.int32)
    has_elim = np.zeros(n_weeks, dtype=bool)  # 标记哪些周有淘汰

    # 规则和分数
    rule_method = np.zeros(n_weeks, dtype=np.int32)
    judge_save_active = np.zeros(n_weeks, dtype=bool)  # 评委拯救是否激活
    judge_pct_padded = np.zeros((n_weeks, max_contestants), dtype=np.float32)
    judge_rank_padded = np.zeros((n_weeks, max_contestants), dtype=np.float32)

    # 反向映射: obs_idx -> (week, position) 用于散射
    obs_to_week = np.zeros(n_obs, dtype=np.int32)
    obs_to_pos = np.zeros(n_obs, dtype=np.int32)

    for w, wd in enumerate(week_data):
        indices = np.where(wd['obs_mask'])[0]
        n_cont = len(indices)
        week_indices[w, :n_cont] = indices
        week_mask[w, :n_cont] = True
        rule_method[w] = wd['rule_method']
        judge_save_active[w] = wd.get('judge_save_active', False)
        judge_pct_padded[w, :n_cont] = judge_score_pct[indices]
        judge_rank_padded[w, :n_cont] = judge_rank_score[indices]

        # 反向映射
        for pos, obs_idx in enumerate(indices):
            obs_to_week[obs_idx] = w
            obs_to_pos[obs_idx] = pos

        # 淘汰信息
        n_elim = wd['n_eliminated']
        n_elim_per_week[w] = n_elim
        if n_elim > 0:
            has_elim[w] = True
            elim_global = np.where(wd['eliminated_mask'])[0]
            for i, eg in enumerate(elim_global):
                elim_local[w, i] = np.where(indices == eg)[0][0]
                elim_mask[w, i] = True

    return {
        'week_indices': week_indices,
        'week_mask': week_mask,
        'elim_local': elim_local,
        'elim_mask': elim_mask,
        'n_elim_per_week': n_elim_per_week,
        'has_elim': has_elim,
        'rule_method': rule_method,
        'judge_save_active': judge_save_active,
        'judge_pct_padded': judge_pct_padded,
        'judge_rank_padded': judge_rank_padded,
        'obs_to_week': obs_to_week,
        'obs_to_pos': obs_to_pos,
        'n_weeks': n_weeks,
        'n_obs': n_obs,
        'max_contestants': max_contestants,
        'max_elim': max_elim,
    }


def _model_fn(train_data, prior, model_cfg, X_lin, spline_bases, vec_week, celeb_spline_bases=None, linear_feature_names=None):
    """NumPyro 模型定义 (完全向量化版本)"""
    n_celebs = train_data['n_celebs']
    n_pros = train_data['n_pros']
    n_obs = train_data['n_obs']
    celeb_idx = train_data['celeb_idx']
    pro_idx = train_data['pro_idx']

    # 名人效应 α
    alpha_beta = numpyro.sample('alpha_beta',
        dist.Normal(0, prior['beta_celeb_scale']).expand([train_data['X_celeb'].shape[1]]))
    alpha_sigma = numpyro.sample('alpha_sigma',
        dist.InverseGamma(prior['sigma_celeb']['a'], prior['sigma_celeb']['b']))

    # 名人效应噪声：Normal 或 Student-t
    if model_cfg.get('use_student_t_effects', False):
        celeb_df_fixed = model_cfg.get('celeb_df')
        if celeb_df_fixed is None:
            alpha_df = numpyro.sample('alpha_df',
                dist.Gamma(prior['celeb_df_prior']['a'], prior['celeb_df_prior']['b']))
            alpha_df = alpha_df + 2.0
        else:
            alpha_df = celeb_df_fixed
        alpha_eps = numpyro.sample('alpha_eps',
            dist.StudentT(df=alpha_df, loc=0, scale=1).expand([n_celebs]))
    else:
        alpha_eps = numpyro.sample('alpha_eps', dist.Normal(0, 1).expand([n_celebs]))

    # 名人级样条贡献
    celeb_spline_contrib = 0.0
    if celeb_spline_bases is not None:
        for i, basis in enumerate(celeb_spline_bases):
            coef = numpyro.sample(f'celeb_spline_{i}_coef',
                dist.Normal(0, prior['spline_smoothing_scale']).expand([basis.shape[1]]))
            celeb_spline_contrib = celeb_spline_contrib + jnp.dot(basis, coef)

    alpha_raw = jnp.dot(train_data['X_celeb'], alpha_beta) + alpha_sigma * alpha_eps + celeb_spline_contrib
    alpha = numpyro.deterministic('alpha', alpha_raw - jnp.mean(alpha_raw))

    # 舞者效应 δ
    delta_sigma = numpyro.sample('delta_sigma',
        dist.InverseGamma(prior['sigma_pro']['a'], prior['sigma_pro']['b']))

    # 舞者效应噪声：Normal 或 Student-t
    if model_cfg.get('use_student_t_effects', False):
        pro_df_fixed = model_cfg.get('pro_df')
        if pro_df_fixed is None:
            delta_df = numpyro.sample('delta_df',
                dist.Gamma(prior['pro_df_prior']['a'], prior['pro_df_prior']['b']))
            delta_df = delta_df + 2.0
        else:
            delta_df = pro_df_fixed
        delta_eps = numpyro.sample('delta_eps',
            dist.StudentT(df=delta_df, loc=0, scale=1).expand([n_pros]))
    else:
        delta_eps = numpyro.sample('delta_eps', dist.Normal(0, 1).expand([n_pros]))

    # 舞者特征线性部分（可选）
    if model_cfg.get('use_pro_features', True) and train_data['X_pro'].shape[1] > 0:
        delta_beta = numpyro.sample('delta_beta',
            dist.Normal(0, prior['beta_pro_scale']).expand([train_data['X_pro'].shape[1]]))
        delta_raw = jnp.dot(train_data['X_pro'], delta_beta) + delta_sigma * delta_eps
    else:
        delta_raw = delta_sigma * delta_eps

    delta = numpyro.deterministic('delta', delta_raw - jnp.mean(delta_raw))

    # 线性特征贡献
    if X_lin is not None and X_lin.shape[1] > 0:
        n_features = X_lin.shape[1]

        # 识别需要正向先验的特征
        positive_prior_features = model_cfg.get('positive_prior_features') or []
        positive_mask = jnp.array([
            name in positive_prior_features
            for name in (linear_feature_names or [])
        ])

        if model_cfg.get('use_horseshoe', False):
            # Regularized Horseshoe 先验（非中心化参数化）
            # 参考: Piironen & Vehtari (2017), Betancourt (2017)

            # 从配置读取Horseshoe参数（可调节收缩强度）
            tau_scale = model_cfg.get('horseshoe_tau_scale', 1.0)  # 全局收缩尺度
            c2_scale = model_cfg.get('horseshoe_c2_scale', 1.0)    # Slab方差尺度

            # 全局收缩参数 τ（尺度越大，收缩越弱）
            tau_hs = numpyro.sample('tau_hs', dist.HalfCauchy(tau_scale))

            # 局部收缩参数 λ_j（非中心化）
            lambda_hs_raw = numpyro.sample('lambda_hs_raw', dist.HalfNormal(1.0).expand([n_features]))
            lambda_hs = numpyro.deterministic('lambda_hs', lambda_hs_raw)

            # Slab 方差 c² (限制大系数的最大值，scale越大允许更大系数)
            c2_hs = numpyro.sample('c2_hs', dist.InverseGamma(2.0, c2_scale))

            # Regularized 局部方差: λ̃² = c² * λ² / (c² + τ² * λ²)
            lambda2 = lambda_hs ** 2
            tau2 = tau_hs ** 2
            lambda_tilde2 = c2_hs * lambda2 / (c2_hs + tau2 * lambda2)
            scale = tau_hs * jnp.sqrt(lambda_tilde2)

            # 非中心化参数化：beta = scale * beta_raw
            beta_raw = numpyro.sample('beta_raw', dist.Normal(0, 1).expand([n_features]))
            beta_obs_raw = scale * beta_raw
            # 对正向先验特征取绝对值
            beta_obs = numpyro.deterministic('beta_obs',
                jnp.where(positive_mask, jnp.abs(beta_obs_raw), beta_obs_raw))
        else:
            # 向量化采样，然后对正向先验特征取绝对值
            beta_obs_raw = numpyro.sample('beta_obs_raw',
                dist.Normal(0, prior['beta_obs_scale']).expand([n_features]))
            beta_obs = numpyro.deterministic('beta_obs',
                jnp.where(positive_mask, jnp.abs(beta_obs_raw), beta_obs_raw))

        lin_contrib = jnp.dot(X_lin, beta_obs)
    else:
        lin_contrib = 0.0

    # 样条特征贡献
    spline_contrib = 0.0
    for i, basis in enumerate(spline_bases):
        coef = numpyro.sample(f'spline_{i}_coef',
            dist.Normal(0, prior['spline_smoothing_scale']).expand([basis.shape[1]]))
        spline_contrib = spline_contrib + jnp.dot(basis, coef)

    # 潜在投票强度 μ
    mu_mean = alpha[celeb_idx] + delta[pro_idx] + lin_contrib + spline_contrib

    # 观测级噪声（Student-t 分布）
    if model_cfg.get('obs_noise', False):
        # 噪声尺度
        sigma_obs = numpyro.sample('sigma_obs',
            dist.InverseGamma(prior['sigma_obs']['a'], prior['sigma_obs']['b']))

        # 自由度（可学习或固定）
        obs_df_fixed = model_cfg.get('obs_noise_df')
        if obs_df_fixed is None:
            # 学习自由度，Gamma 先验
            obs_df = numpyro.sample('obs_df',
                dist.Gamma(prior['obs_df_prior']['a'], prior['obs_df_prior']['b']))
            obs_df = obs_df + 2.0  # 确保 df > 2，方差有限
        else:
            obs_df = obs_df_fixed

        # Student-t 噪声
        obs_eps = numpyro.sample('obs_eps',
            dist.StudentT(df=obs_df, loc=0, scale=1).expand([n_obs]))
        mu = mu_mean + sigma_obs * obs_eps
    else:
        mu = mu_mean

    numpyro.deterministic('mu', mu)

    # === 向量化: 按周分组线性归一化计算 P_fan ===
    week_indices = vec_week['week_indices']  # [n_weeks, max_contestants]
    week_mask = vec_week['week_mask']        # [n_weeks, max_contestants]
    obs_to_week = vec_week['obs_to_week']    # [n_obs]
    obs_to_pos = vec_week['obs_to_pos']      # [n_obs]

    # 用0填充无效索引，取mu值后用mask处理
    safe_indices = jnp.where(week_indices >= 0, week_indices, 0)
    mu_padded = mu[safe_indices]  # [n_weeks, max_contestants]

    # === P_fan 归一化方式选择 ===
    normalize_method = model_cfg.get('p_fan_normalize', 'softmax')  # 'linear' or 'softmax'

    if normalize_method == 'softmax':
        # Softmax 归一化: P_fan = softmax(mu / T_fan)
        # 温度参数 T_fan 控制平滑度：T 大 → 更均匀，T 小 → 更尖锐
        if model_cfg.get('learn_t_fan', False):
            T_fan = numpyro.sample('T_fan', dist.HalfNormal(model_cfg.get('t_fan_init', 1.0)))
        else:
            T_fan = model_cfg.get('t_fan_init', 1.0)

        # 数值稳定的 softmax：减去最大值
        mu_masked = jnp.where(week_mask, mu_padded, -jnp.inf)  # 无效位置设为-inf
        mu_max = jnp.max(mu_masked, axis=1, keepdims=True)  # [n_weeks, 1]
        mu_shifted = mu_padded - mu_max  # 数值稳定
        exp_mu = jnp.exp(mu_shifted / T_fan)
        exp_mu = jnp.where(week_mask, exp_mu, 0.0)  # 无效位置为0
        exp_sum = jnp.sum(exp_mu, axis=1, keepdims=True) + 1e-10
        P_fan_padded = exp_mu / exp_sum
    else:
        # 线性归一化: P_fan = (mu - min) / sum(mu - min)
        mu_masked = jnp.where(week_mask, mu_padded, jnp.inf)  # 无效位置设为inf，不影响min
        mu_min = jnp.min(mu_masked, axis=1, keepdims=True)  # [n_weeks, 1]
        mu_shifted = jnp.where(week_mask, mu_padded - mu_min, 0.0)  # 平移到非负
        mu_sum = jnp.sum(mu_shifted, axis=1, keepdims=True) + 1e-10  # 防止除零
        P_fan_padded = mu_shifted / mu_sum  # [n_weeks, max_contestants]
        P_fan_padded = jnp.where(week_mask, P_fan_padded, 0.0)

    # 向量化散射: 使用预计算的反向映射
    P_fan = P_fan_padded[obs_to_week, obs_to_pos]
    numpyro.deterministic('P_fan', P_fan)

    # === 向量化: 综合得分 S ===
    rule_method = vec_week['rule_method']  # [n_weeks]
    judge_pct = vec_week['judge_pct_padded']   # [n_weeks, max_contestants]
    judge_rank = vec_week['judge_rank_padded'] # [n_weeks, max_contestants]

    # 百分比法: S = judge_pct + P_fan
    S_pct = judge_pct + P_fan_padded

    # 排名法: soft_rank
    n_contestants_per_week = week_mask.sum(axis=1, keepdims=True)  # [n_weeks, 1]
    diff = P_fan_padded[:, :, None] - P_fan_padded[:, None, :]  # [n_weeks, max, max]
    soft_rank = jnp.sum(jax.nn.sigmoid(diff / 0.1) * week_mask[:, None, :], axis=2) - 0.5
    R_fan = soft_rank / jnp.maximum(n_contestants_per_week - 1, 1)
    S_rank = judge_rank + R_fan

    # 根据rule选择
    is_pct = (rule_method == 1)[:, None]  # [n_weeks, 1]
    S_padded = jnp.where(is_pct, S_pct, S_rank)
    S_padded = jnp.where(week_mask, S_padded, 0.0)

    # 向量化散射: 使用预计算的反向映射
    S = S_padded[obs_to_week, obs_to_pos]
    numpyro.deterministic('S', S)

    # 似然函数温度参数 τ
    if model_cfg['learn_tau']:
        tau = numpyro.sample('tau', dist.HalfNormal(model_cfg['tau_init']))
    else:
        tau = model_cfg['tau_init']

    # === 评委拯救修正参数 θ_save ===
    # θ_save > 0 表示评委倾向于保留技术分更高的选手
    if model_cfg.get('use_judge_save_correction', True):
        theta_save = numpyro.sample('theta_save', dist.HalfNormal(model_cfg.get('theta_save_scale', 1.0)))
    else:
        theta_save = 0.0

    # === 向量化 Plackett-Luce 似然（使用 jax.lax.cond 条件分支优化） ===
    elim_local = vec_week['elim_local']      # [n_weeks, max_elim]
    elim_mask = vec_week['elim_mask']        # [n_weeks, max_elim]
    has_elim = vec_week['has_elim']          # [n_weeks]
    judge_save_active = vec_week['judge_save_active']  # [n_weeks]
    max_elim = vec_week['max_elim']

    # 获取评委分（用于 judge_save 修正）
    judge_score_padded = vec_week['judge_pct_padded']  # [n_weeks, max_contestants]

    # 计算 neg_tau_S
    neg_tau_S_padded = -tau * S_padded  # [n_weeks, max_contestants]

    # === 简单版本：无评委拯救的周（不需要排序和修正） ===
    def pl_simple(neg_tau_S_w, valid_contestants, elim_w, elim_valid_w):
        """标准 Plackett-Luce 对数似然（无评委拯救修正）"""
        def scan_fn(remaining, i):
            e_idx = elim_w[i]
            is_valid = elim_valid_w[i]
            log_remaining = jnp.log(remaining + 1e-10)
            log_denom = jax.scipy.special.logsumexp(neg_tau_S_w + log_remaining)
            ll = neg_tau_S_w[e_idx] - log_denom
            ll = jnp.where(is_valid, ll, 0.0)
            new_remaining = jnp.where(
                is_valid,
                remaining.at[e_idx].set(0.0),
                remaining
            )
            return new_remaining, ll

        init_remaining = valid_contestants.astype(jnp.float32)
        _, lls = jax.lax.scan(scan_fn, init_remaining, jnp.arange(max_elim))
        return lls.sum()

    # === 复杂版本：有评委拯救的周（需要排序和修正） ===
    def pl_with_judge_save(neg_tau_S_w, S_w, judge_w, valid_contestants, elim_w, elim_valid_w):
        """带评委拯救修正的 Plackett-Luce 对数似然"""
        # 找出 S 最低的两人（危险区）
        S_for_sort = jnp.where(valid_contestants, S_w, jnp.inf)
        bottom2_idx = jnp.argsort(S_for_sort)[:2]

        # 计算危险区两人的评委分均值
        judge_bottom2 = judge_w[bottom2_idx]
        judge_mean_bottom2 = jnp.mean(judge_bottom2)

        # 修正后的 S: S' = S + θ_save * (J - J_mean_bottom2)
        judge_diff = judge_w - judge_mean_bottom2
        S_corrected = S_w + theta_save * judge_diff
        neg_tau_S_corrected = -tau * S_corrected

        # 构建危险区 mask（只有 bottom2 的两人）
        bottom2_mask = jnp.zeros_like(valid_contestants)
        bottom2_mask = bottom2_mask.at[bottom2_idx[0]].set(True)
        bottom2_mask = bottom2_mask.at[bottom2_idx[1]].set(True)
        bottom2_mask = bottom2_mask & valid_contestants

        # Plackett-Luce 似然（只在危险区两人中选）
        def scan_fn(remaining, i):
            e_idx = elim_w[i]
            is_valid = elim_valid_w[i]
            log_remaining = jnp.log(remaining + 1e-10)
            log_denom = jax.scipy.special.logsumexp(neg_tau_S_corrected + log_remaining)
            ll = neg_tau_S_corrected[e_idx] - log_denom
            ll = jnp.where(is_valid, ll, 0.0)
            new_remaining = jnp.where(
                is_valid,
                remaining.at[e_idx].set(0.0),
                remaining
            )
            return new_remaining, ll

        init_remaining = bottom2_mask.astype(jnp.float32)
        _, lls = jax.lax.scan(scan_fn, init_remaining, jnp.arange(max_elim))
        return lls.sum()

    # === 合并版本：使用 jax.lax.cond 条件分支，避免重复计算 ===
    def pl_combined(neg_tau_S_w, S_w, judge_w, valid_contestants, elim_w, elim_valid_w,
                    has_elim_w, judge_save_active_w):
        """根据条件选择执行哪个似然函数，避免两次计算"""
        def compute_with_save():
            return pl_with_judge_save(neg_tau_S_w, S_w, judge_w, valid_contestants, elim_w, elim_valid_w)

        def compute_simple():
            return pl_simple(neg_tau_S_w, valid_contestants, elim_w, elim_valid_w)

        def compute_ll():
            return jax.lax.cond(judge_save_active_w, compute_with_save, compute_simple)

        # 如果没有淘汰，返回0；否则根据 judge_save 条件选择计算方式
        return jax.lax.cond(has_elim_w, compute_ll, lambda: 0.0)

    # 单次 vmap，每周只计算一次似然
    lls = jax.vmap(pl_combined)(
        neg_tau_S_padded,      # [n_weeks, max_contestants]
        S_padded,              # [n_weeks, max_contestants]
        judge_score_padded,    # [n_weeks, max_contestants]
        week_mask,             # [n_weeks, max_contestants]
        elim_local,            # [n_weeks, max_elim]
        elim_mask,             # [n_weeks, max_elim]
        has_elim,              # [n_weeks]
        judge_save_active      # [n_weeks]
    )

    total_ll = jnp.sum(lls)
    numpyro.factor('likelihood', total_ll)


def _parse_interaction(expr, X_obs_orig, X_obs_names_orig):
    """解析交互项表达式并计算新特征"""
    import re
    expr = expr.strip()

    # 匹配 "feat1 * (1 - feat2)" 格式
    match = re.match(r'(\w+)\s*\*\s*\(1\s*-\s*(\w+)\)', expr)
    if match:
        feat1, feat2 = match.groups()
        idx1 = X_obs_names_orig.index(feat1)
        idx2 = X_obs_names_orig.index(feat2)
        new_col = X_obs_orig[:, idx1] * (1 - X_obs_orig[:, idx2])
        new_name = f"{feat1}_x_not_{feat2}"
        return new_col, new_name

    # 匹配 "feat1 * feat2" 格式
    match = re.match(r'(\w+)\s*\*\s*(\w+)', expr)
    if match:
        feat1, feat2 = match.groups()
        idx1 = X_obs_names_orig.index(feat1)
        idx2 = X_obs_names_orig.index(feat2)
        new_col = X_obs_orig[:, idx1] * X_obs_orig[:, idx2]
        new_name = f"{feat1}_x_{feat2}"
        return new_col, new_name

    raise ValueError(f"无法解析交互项表达式: {expr}")


def _process_celeb_features(train_data, model_cfg):
    """处理名人特征：交互项和排除"""
    X_celeb_orig = train_data['X_celeb'].copy()
    X_celeb_names_orig = list(train_data['X_celeb_names'])

    X_celeb = X_celeb_orig.copy()
    X_celeb_names = list(X_celeb_names_orig)

    celeb_interaction = model_cfg.get('celeb_interaction_features') or []
    exclude_celeb = model_cfg.get('exclude_celeb_features') or []

    # 添加交互项
    if celeb_interaction:
        print(f"  [build_model] Adding celeb interaction features:")
        for expr in celeb_interaction:
            new_col, new_name = _parse_interaction(expr, X_celeb_orig, X_celeb_names_orig)
            X_celeb = np.column_stack([X_celeb, new_col])
            X_celeb_names.append(new_name)
            print(f"    + {new_name} (from: {expr})")

    # 排除特征
    if exclude_celeb:
        keep_cols = [i for i, name in enumerate(X_celeb_names) if name not in exclude_celeb]
        X_celeb = X_celeb[:, keep_cols]
        X_celeb_names = [X_celeb_names[i] for i in keep_cols]
        print(f"  [build_model] Excluded celeb features: {exclude_celeb}")
        print(f"  [build_model] Remaining celeb features ({len(X_celeb_names)}): {X_celeb_names}")

    return X_celeb.astype(np.float32), X_celeb_names


def build_model(config, datas):
    """构建模型参数（NumPyro 不需要预构建模型对象）"""
    train_data = datas['train_data']
    model_cfg = config['model']

    # === 处理名人特征 ===
    X_celeb, X_celeb_names = _process_celeb_features(train_data, model_cfg)

    # === 处理观测特征 ===
    X_obs_orig = train_data['X_obs'].copy()
    X_obs_names_orig = list(train_data['X_obs_names'])

    X_obs = X_obs_orig.copy()
    X_obs_names = list(X_obs_names_orig)
    spline_features = model_cfg['spline_features'] or []
    exclude_obs = model_cfg.get('exclude_obs_features') or []
    obs_interaction = model_cfg.get('obs_interaction_features') or []
    center_features = model_cfg.get('center_features', False)

    # 添加观测特征交互项（在排除之前，使用原始特征）
    if obs_interaction:
        print(f"  [build_model] Adding obs interaction features:")
        for expr in obs_interaction:
            new_col, new_name = _parse_interaction(expr, X_obs_orig, X_obs_names_orig)
            X_obs = np.column_stack([X_obs, new_col])
            X_obs_names.append(new_name)
            print(f"    + {new_name} (from: {expr})")

    # 过滤掉排除的观测特征
    if exclude_obs:
        keep_cols = [i for i, name in enumerate(X_obs_names) if name not in exclude_obs]
        X_obs = X_obs[:, keep_cols]
        X_obs_names = [X_obs_names[i] for i in keep_cols]
        print(f"  [build_model] Excluded obs features: {exclude_obs}")
        print(f"  [build_model] Remaining obs features ({len(X_obs_names)}): {X_obs_names}")

    # 特征中心化
    feature_means = None
    if center_features:
        feature_means = X_obs.mean(axis=0)
        X_obs = X_obs - feature_means
        print(f"  [build_model] Centered {len(X_obs_names)} obs features")

    # 分离线性/样条特征（观测级）
    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    X_lin = X_obs[:, linear_cols] if linear_cols else None
    linear_feature_names = [X_obs_names[c] for c in linear_cols]
    spline_bases = [_build_spline_basis(X_obs[:, c], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree']) for c in spline_cols]
    spline_feature_names = [X_obs_names[c] for c in spline_cols]
    if spline_features:
        print(f"  [build_model] Obs spline features: {spline_feature_names}")

    # 打印正向先验特征
    positive_prior_features = model_cfg.get('positive_prior_features') or []
    if positive_prior_features:
        active_positive = [f for f in positive_prior_features if f in linear_feature_names]
        if active_positive:
            print(f"  [build_model] Positive prior features: {active_positive}")

    # === 处理名人级样条特征 ===
    celeb_spline_features = model_cfg.get('celeb_spline_features') or []
    X_celeb_full = train_data['X_celeb']  # 原始名人特征
    X_celeb_names_full = train_data['X_celeb_names']
    celeb_spline_bases = []
    celeb_spline_feature_names = []
    if celeb_spline_features:
        print(f"  [build_model] Building celeb spline features:")
        for feat_name in celeb_spline_features:
            if feat_name in X_celeb_names_full:
                col_idx = X_celeb_names_full.index(feat_name)
                basis = _build_spline_basis(X_celeb_full[:, col_idx],
                                           model_cfg['n_spline_knots'], model_cfg['spline_degree'])
                celeb_spline_bases.append(basis)
                celeb_spline_feature_names.append(feat_name)
                print(f"    + {feat_name} spline basis: {basis.shape}")
            else:
                print(f"    ! {feat_name} not found in X_celeb_names")

    # 构建向量化week数据
    vec_week = _build_vectorized_week_data(
        train_data['week_data'],
        train_data['n_obs'],
        train_data['judge_score_pct'],
        train_data['judge_rank_score']
    )

    # 更新 train_data 中的特征（用于模型和后续分析）
    train_data_updated = dict(train_data)
    # 名人特征（含交互项、排除后）
    train_data_updated['X_celeb'] = X_celeb
    train_data_updated['X_celeb_names'] = X_celeb_names
    # 观测特征（含交互项、排除后、可能已中心化）
    train_data_updated['X_obs'] = X_obs.astype(np.float32)
    train_data_updated['X_obs_names'] = X_obs_names

    # 保存分离后的线性/样条特征（用于训练对称的线性模型）
    train_data_updated['X_lin'] = X_lin  # 线性特征矩阵 [n_obs, n_linear_features]
    train_data_updated['linear_feature_names'] = linear_feature_names
    train_data_updated['spline_bases'] = spline_bases  # 样条基矩阵列表 [basis_i: [n_obs, n_basis]]
    train_data_updated['spline_feature_names'] = spline_feature_names
    train_data_updated['celeb_spline_bases'] = celeb_spline_bases if celeb_spline_bases else None
    train_data_updated['celeb_spline_feature_names'] = celeb_spline_feature_names
    train_data_updated['feature_means'] = feature_means  # 中心化均值（如果启用）

    return {
        'train_data': train_data_updated,
        'prior': config['prior'],
        'model_cfg': model_cfg,
        'X_lin': X_lin,
        'linear_feature_names': linear_feature_names,
        'spline_bases': spline_bases,
        'spline_feature_names': spline_feature_names,
        'celeb_spline_bases': celeb_spline_bases if celeb_spline_bases else None,
        'celeb_spline_feature_names': celeb_spline_feature_names,
        'vec_week': vec_week,
        'feature_means': feature_means,
        'X_obs_names': X_obs_names,
        'X_obs_names_orig': X_obs_names_orig,
        'X_celeb_names': X_celeb_names,
    }


def train(config, model_params, datas):
    """MCMC采样，trace 存入 datas['trace']"""
    import sys
    samp = config['sampling']

    print(f"    [train] Starting MCMC: {samp['n_chains']} chains, {samp['n_tune']} tune, {samp['n_samples']} samples")
    sys.stdout.flush()

    # 创建 NUTS 采样器
    kernel = NUTS(_model_fn, target_accept_prob=samp['target_accept'])
    mcmc = MCMC(kernel,
                num_warmup=samp['n_tune'],
                num_samples=samp['n_samples'],
                num_chains=samp['n_chains'],
                progress_bar=True)

    # 运行采样
    rng_key = jax.random.PRNGKey(samp['random_seed'])
    mcmc.run(rng_key,
             train_data=model_params['train_data'],
             prior=model_params['prior'],
             model_cfg=model_params['model_cfg'],
             X_lin=model_params['X_lin'],
             spline_bases=model_params['spline_bases'],
             vec_week=model_params['vec_week'],
             celeb_spline_bases=model_params.get('celeb_spline_bases'),
             linear_feature_names=model_params.get('linear_feature_names'))

    print("    [train] MCMC completed")
    datas['trace'] = mcmc.get_samples()
    # 保存特征处理信息用于预测
    datas['feature_means'] = model_params.get('feature_means')
    datas['X_obs_names_filtered'] = model_params.get('X_obs_names')
    datas['celeb_spline_bases'] = model_params.get('celeb_spline_bases')
    datas['celeb_spline_feature_names'] = model_params.get('celeb_spline_feature_names')
    datas['spline_feature_names'] = model_params.get('spline_feature_names')
    return datas


def extract_posterior(config, datas):
    """从 trace 提取后验样本"""
    samples = datas['trace']

    # NumPyro 的 samples 已经是 dict 格式
    datas['posterior_samples'] = {
        'alpha': np.array(samples['alpha']),
        'delta': np.array(samples['delta']),
    }

    if 'beta_obs' in samples:
        datas['posterior_samples']['beta_obs'] = np.array(samples['beta_obs'])

    # 观测级样条系数
    for i in range(20):
        key = f'spline_{i}_coef'
        if key in samples:
            datas['posterior_samples'][key] = np.array(samples[key])
        else:
            break

    # 名人级样条系数
    for i in range(20):
        key = f'celeb_spline_{i}_coef'
        if key in samples:
            datas['posterior_samples'][key] = np.array(samples[key])
        else:
            break

    if 'tau' in samples:
        datas['posterior_samples']['tau'] = np.array(samples['tau'])

    return datas


def compute_metrics(config, datas):
    """计算评估指标"""
    S_samples = datas['S_samples']  # [n_samples, n_obs]
    P_fan_samples = datas.get('P_fan_samples')  # [n_samples, n_obs]
    week_data = datas['train_data']['week_data']

    n_samples = S_samples.shape[0]
    S_mean = S_samples.mean(axis=0)  # 后验均值
    week_results = []

    for wd in week_data:
        k = wd['n_eliminated']
        season = wd['season']
        week = wd['week']

        # 赛季类型分类
        if season <= 10:
            season_era = 'Early'
        elif season <= 20:
            season_era = 'Middle'
        else:
            season_era = 'Late'

        if k == 0:
            week_results.append({
                'accuracy': None, 'accuracy_mean': None,
                'season': season, 'week': week, 'season_era': season_era,
                'p_fan_var': None, 'decision_gap': None,
            })
            continue

        mask = wd['obs_mask']
        indices = np.where(mask)[0]

        elim_global = np.where(wd['eliminated_mask'])[0]
        actual_elim = set(np.searchsorted(indices, elim_global))

        # 1. 按后验样本计算准确率 (Acc_B) - 按淘汰次数计算
        correct_elim_count = 0  # 正确预测的淘汰次数（所有样本累计）
        total_elim_count = k * n_samples  # 总淘汰次数 = k * n_samples
        for s in range(n_samples):
            S_week = S_samples[s, mask]

            if wd['judge_save_active']:
                pred_danger = set(np.argsort(S_week)[:2])
                # 每个实际淘汰者是否在危险区内
                correct_elim_count += len(actual_elim & pred_danger)
            else:
                pred_elim = set(np.argsort(S_week)[:k])
                # 每个实际淘汰者是否被正确预测
                correct_elim_count += len(actual_elim & pred_elim)

        accuracy = correct_elim_count / total_elim_count  # 淘汰预测准确率

        # 2. 按后验均值计算准确率 (Acc_A) - 按淘汰次数计算
        S_week_mean = S_mean[mask]
        if wd['judge_save_active']:
            pred_danger_mean = set(np.argsort(S_week_mean)[:2])
            correct_elim_mean = len(actual_elim & pred_danger_mean)
        else:
            pred_elim_mean = set(np.argsort(S_week_mean)[:k])
            correct_elim_mean = len(actual_elim & pred_elim_mean)

        accuracy_mean = correct_elim_mean / k  # 淘汰预测准确率（点估计）

        # 3. 决策缺口 δ_i = |f(S̄) - p_i|
        decision_gap = abs(accuracy_mean - accuracy)

        # 4. P_fan 后验样本方差（该周所有选手的平均方差）
        p_fan_var = None
        if P_fan_samples is not None:
            P_fan_week = P_fan_samples[:, mask]  # [n_samples, n_contestants]
            p_fan_var = float(np.mean(np.var(P_fan_week, axis=0)))

        week_results.append({
            'accuracy': accuracy,           # p_i (Acc_B per week, 按淘汰次数)
            'accuracy_mean': accuracy_mean, # f(S̄) (Acc_A per week, 按淘汰次数)
            'decision_gap': decision_gap,   # δ_i
            'p_fan_var': p_fan_var,         # 后验样本方差
            'season': season,
            'week': week,
            'season_era': season_era,
            'judge_save': wd['judge_save_active'],
            'n_contestants': int(mask.sum()),
            'n_eliminated': k,
            'correct_elim_mean': correct_elim_mean,  # 点估计正确淘汰数
        })

    valid_results = [r for r in week_results if r['accuracy'] is not None]
    valid_gap = [r['decision_gap'] for r in valid_results]
    valid_var = [r['p_fan_var'] for r in valid_results if r['p_fan_var'] is not None]

    # 按淘汰次数加权计算全局准确率
    total_elim = sum(r['n_eliminated'] for r in valid_results)
    total_correct_mean = sum(r['correct_elim_mean'] for r in valid_results)
    # Acc_B: 按淘汰次数加权平均
    weighted_acc_b = sum(r['accuracy'] * r['n_eliminated'] for r in valid_results) / total_elim if total_elim > 0 else 0.0
    # Acc_A: 总正确淘汰数 / 总淘汰数
    weighted_acc_a = total_correct_mean / total_elim if total_elim > 0 else 0.0

    datas['metrics'] = {
        'week_results': week_results,
        'mean_accuracy': weighted_acc_b,              # 全局 Acc_B (按淘汰次数加权)
        'mean_accuracy_expectation': weighted_acc_a,  # 全局 Acc_A (按淘汰次数加权)
        'mean_decision_gap': np.mean(valid_gap) if valid_gap else 0.0,
        'mean_p_fan_var': np.mean(valid_var) if valid_var else 0.0,
        'n_weeks_evaluated': len(valid_results),
        'n_eliminations': total_elim,  # 总淘汰次数
    }

    print(f"Elimination accuracy (Acc_B, posterior): {datas['metrics']['mean_accuracy']:.3f} ({total_elim} eliminations)")
    print(f"Elimination accuracy (Acc_A, point est): {datas['metrics']['mean_accuracy_expectation']:.3f}")
    print(f"Mean decision gap (|Acc_A - Acc_B|):     {datas['metrics']['mean_decision_gap']:.3f}")
    print(f"Mean P_fan variance:                     {datas['metrics']['mean_p_fan_var']:.6f}")
    return datas


def generate_output(config, datas):
    """生成模型输出，按 MODEL_OUTPUT.md 规范"""
    from scipy.special import softmax

    posterior = datas['posterior_samples']
    td = datas['train_data']
    model_cfg = config['model']

    alpha = posterior['alpha'].mean(axis=0)
    delta = posterior['delta'].mean(axis=0)

    celeb_idx = td['celeb_idx']
    pro_idx = td['pro_idx']
    week_idx = td['week_idx']
    n_obs = td['n_obs']

    # 直接使用 train_data 中已处理好的特征（build_model 已处理交互项、排除、中心化）
    X_lin = td.get('X_lin')
    linear_names = td.get('linear_feature_names', [])
    spline_bases = td.get('spline_bases', [])
    spline_names = td.get('spline_feature_names', [])

    alpha_contrib = alpha[celeb_idx]
    delta_contrib = delta[pro_idx]

    linear_contrib = np.zeros(n_obs, dtype=np.float32)
    linear_details = {}
    if X_lin is not None and 'beta_obs' in posterior:
        beta_obs = posterior['beta_obs'].mean(axis=0)
        linear_contrib = X_lin @ beta_obs
        for i, name in enumerate(linear_names):
            linear_details[f'linear_{name}'] = X_lin[:, i] * beta_obs[i]

    spline_contrib = np.zeros(n_obs, dtype=np.float32)
    spline_details = {}
    spline_coefs = []
    for i, basis in enumerate(spline_bases):
        key = f'spline_{i}_coef'
        if key in posterior:
            coef = posterior[key].mean(axis=0)
            spline_coefs.append(coef)
            contrib_i = basis @ coef
            spline_contrib = spline_contrib + contrib_i
            spline_details[f'spline_{spline_names[i]}'] = contrib_i

    mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib

    # P_fan 归一化
    normalize_method = model_cfg.get('p_fan_normalize', 'softmax')
    T_fan = model_cfg.get('t_fan_init', 1.0)

    P_fan = np.zeros(n_obs, dtype=np.float32)
    for wd in td['week_data']:
        mask = wd['obs_mask']
        mu_week = mu[mask]

        if normalize_method == 'softmax':
            # Softmax 归一化
            mu_shifted = mu_week - mu_week.max()  # 数值稳定
            exp_mu = np.exp(mu_shifted / T_fan)
            P_fan[mask] = exp_mu / (exp_mu.sum() + 1e-10)
        else:
            # 线性归一化
            mu_shifted = mu_week - mu_week.min()
            P_fan[mask] = mu_shifted / (mu_shifted.sum() + 1e-10)

    output = {
        'n_obs': n_obs,
        'n_celebs': td['n_celebs'],
        'n_pros': td['n_pros'],
        'celeb_idx': celeb_idx,
        'pro_idx': pro_idx,
        'week_idx': week_idx,
        'mu': mu,
        'P_fan': P_fan,
        'alpha_contrib': alpha_contrib,
        'delta_contrib': delta_contrib,
        'linear_contrib': linear_contrib,
        'spline_contrib': spline_contrib,
        'alpha': alpha,
        'delta': delta,
        'beta_obs': posterior.get('beta_obs', np.array([])).mean(axis=0) if 'beta_obs' in posterior else np.array([]),
        'spline_coefs': spline_coefs,
        'feature_names': {
            'beta_obs': linear_names,
            'splines': spline_names,
        },
    }

    output.update(linear_details)
    output.update(spline_details)

    datas['model_output'] = output
    return datas


def predict(config, train_datas, eval_datas):
    """用后验样本对 eval_datas 计算 S 和 P_fan"""
    from scipy.special import softmax
    from tqdm import tqdm

    posterior = train_datas['posterior_samples']
    td = eval_datas['train_data']
    model_cfg = config['model']

    n_samples = posterior['alpha'].shape[0]
    n_obs = td['n_obs']

    # 索引
    celeb_idx = td['celeb_idx']
    pro_idx = td['pro_idx']

    # 直接使用 train_data 中已处理好的特征（build_model 已处理交互项、排除、中心化）
    X_lin = td.get('X_lin')
    spline_bases = td.get('spline_bases', [])
    spline_feature_names = td.get('spline_feature_names', [])

    # 对每个后验样本计算 S 和 P_fan
    S_samples = np.zeros((n_samples, n_obs), dtype=np.float32)
    P_fan_samples = np.zeros((n_samples, n_obs), dtype=np.float32)

    for s in tqdm(range(n_samples), desc="Predicting"):
        # mu = alpha[celeb_idx] + delta[pro_idx] + linear + spline
        mu = posterior['alpha'][s, celeb_idx] + posterior['delta'][s, pro_idx]

        if X_lin is not None and 'beta_obs' in posterior:
            mu = mu + X_lin @ posterior['beta_obs'][s]

        for i, basis in enumerate(spline_bases):
            key = f'spline_{i}_coef'
            if key in posterior:
                mu = mu + basis @ posterior[key][s]

        # P_fan 归一化
        normalize_method = model_cfg.get('p_fan_normalize', 'softmax')
        T_fan = model_cfg.get('t_fan_init', 1.0)
        # 如果学习 T_fan，使用后验样本
        if model_cfg.get('learn_t_fan', False) and 'T_fan' in posterior:
            T_fan = posterior['T_fan'][s]

        P_fan = np.zeros(n_obs, dtype=np.float32)
        for wd in td['week_data']:
            mask = wd['obs_mask']
            mu_week = mu[mask]

            if normalize_method == 'softmax':
                # Softmax 归一化
                mu_shifted = mu_week - mu_week.max()  # 数值稳定
                exp_mu = np.exp(mu_shifted / T_fan)
                P_fan[mask] = exp_mu / (exp_mu.sum() + 1e-10)
            else:
                # 线性归一化
                mu_shifted = mu_week - mu_week.min()
                P_fan[mask] = mu_shifted / (mu_shifted.sum() + 1e-10)

        P_fan_samples[s] = P_fan

        # S = judge_score + fan_score
        S = np.zeros(n_obs, dtype=np.float32)
        for wd in td['week_data']:
            mask = wd['obs_mask']
            if wd['rule_method'] == 1:  # 百分比法
                S[mask] = td['judge_score_pct'][mask] + P_fan[mask]
            else:  # 排名法
                P_week = P_fan[mask]
                n_contestants = len(P_week)
                if n_contestants > 1:
                    diff = P_week[:, None] - P_week[None, :]
                    soft_rank = np.sum(1 / (1 + np.exp(-diff / 0.1)), axis=1) - 0.5
                    R_fan = soft_rank / (n_contestants - 1)
                else:
                    R_fan = np.array([0.5], dtype=np.float32)
                S[mask] = td['judge_rank_score'][mask] + R_fan

        S_samples[s] = S

    eval_datas['S_samples'] = S_samples
    eval_datas['P_fan_samples'] = P_fan_samples
    return eval_datas
