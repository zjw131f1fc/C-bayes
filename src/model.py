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
        'judge_pct_padded': judge_pct_padded,
        'judge_rank_padded': judge_rank_padded,
        'obs_to_week': obs_to_week,
        'obs_to_pos': obs_to_pos,
        'n_weeks': n_weeks,
        'n_obs': n_obs,
        'max_contestants': max_contestants,
        'max_elim': max_elim,
    }


def _model_fn(train_data, prior, model_cfg, X_lin, spline_bases, vec_week):
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
    alpha_eps = numpyro.sample('alpha_eps', dist.Normal(0, 1).expand([n_celebs]))
    alpha_raw = jnp.dot(train_data['X_celeb'], alpha_beta) + alpha_sigma * alpha_eps
    alpha = numpyro.deterministic('alpha', alpha_raw - jnp.mean(alpha_raw))

    # 舞者效应 δ
    delta_beta = numpyro.sample('delta_beta',
        dist.Normal(0, prior['beta_pro_scale']).expand([train_data['X_pro'].shape[1]]))
    delta_sigma = numpyro.sample('delta_sigma',
        dist.InverseGamma(prior['sigma_pro']['a'], prior['sigma_pro']['b']))
    delta_eps = numpyro.sample('delta_eps', dist.Normal(0, 1).expand([n_pros]))
    delta_raw = jnp.dot(train_data['X_pro'], delta_beta) + delta_sigma * delta_eps
    delta = numpyro.deterministic('delta', delta_raw - jnp.mean(delta_raw))

    # 线性特征贡献
    if X_lin is not None and X_lin.shape[1] > 0:
        beta_obs = numpyro.sample('beta_obs',
            dist.Normal(0, prior['beta_obs_scale']).expand([X_lin.shape[1]]))
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
    mu = alpha[celeb_idx] + delta[pro_idx] + lin_contrib + spline_contrib
    numpyro.deterministic('mu', mu)

    # === 向量化: 按周分组 softmax 计算 P_fan ===
    week_indices = vec_week['week_indices']  # [n_weeks, max_contestants]
    week_mask = vec_week['week_mask']        # [n_weeks, max_contestants]
    obs_to_week = vec_week['obs_to_week']    # [n_obs]
    obs_to_pos = vec_week['obs_to_pos']      # [n_obs]

    # 用0填充无效索引，取mu值后用mask处理
    safe_indices = jnp.where(week_indices >= 0, week_indices, 0)
    mu_padded = mu[safe_indices]  # [n_weeks, max_contestants]
    mu_masked = jnp.where(week_mask, mu_padded, -jnp.inf)
    P_fan_padded = jax.nn.softmax(mu_masked, axis=1)  # [n_weeks, max_contestants]
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

    # === 完全向量化: Plackett-Luce 似然 ===
    elim_local = vec_week['elim_local']      # [n_weeks, max_elim]
    elim_mask = vec_week['elim_mask']        # [n_weeks, max_elim]
    has_elim = vec_week['has_elim']          # [n_weeks]
    max_elim = vec_week['max_elim']

    # 计算 neg_tau_S
    neg_tau_S_padded = -tau * S_padded  # [n_weeks, max_contestants]

    def pl_single_week(neg_tau_S_w, valid_contestants, elim_w, elim_valid_w):
        """单周的Plackett-Luce对数似然 (用于vmap)"""
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

    # 使用 vmap 并行计算所有周的似然
    all_week_lls = jax.vmap(pl_single_week)(
        neg_tau_S_padded,  # [n_weeks, max_contestants]
        week_mask,         # [n_weeks, max_contestants]
        elim_local,        # [n_weeks, max_elim]
        elim_mask          # [n_weeks, max_elim]
    )

    # 只对有淘汰的周求和
    total_ll = jnp.sum(jnp.where(has_elim, all_week_lls, 0.0))

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


def build_model(config, datas):
    """构建模型参数（NumPyro 不需要预构建模型对象）"""
    train_data = datas['train_data']
    model_cfg = config['model']

    # 获取原始特征（用于构造交互项）
    X_obs_orig = train_data['X_obs'].copy()
    X_obs_names_orig = list(train_data['X_obs_names'])

    # 获取特征配置
    X_obs = X_obs_orig.copy()
    X_obs_names = list(X_obs_names_orig)
    spline_features = model_cfg['spline_features'] or []
    exclude_features = model_cfg.get('exclude_features') or []
    interaction_features = model_cfg.get('interaction_features') or []
    center_features = model_cfg.get('center_features', False)

    # 添加交互项（在排除特征之前，使用原始特征计算）
    if interaction_features:
        print(f"  [build_model] Adding interaction features:")
        for expr in interaction_features:
            new_col, new_name = _parse_interaction(expr, X_obs_orig, X_obs_names_orig)
            X_obs = np.column_stack([X_obs, new_col])
            X_obs_names.append(new_name)
            print(f"    + {new_name} (from: {expr})")

    # 过滤掉排除的特征
    if exclude_features:
        keep_cols = [i for i, name in enumerate(X_obs_names) if name not in exclude_features]
        X_obs = X_obs[:, keep_cols]
        X_obs_names = [X_obs_names[i] for i in keep_cols]
        print(f"  [build_model] Excluded features: {exclude_features}")
        print(f"  [build_model] Remaining features ({len(X_obs_names)}): {X_obs_names}")

    # 特征中心化
    feature_means = None
    if center_features:
        feature_means = X_obs.mean(axis=0)
        X_obs = X_obs - feature_means
        print(f"  [build_model] Centered {len(X_obs_names)} features")

    # 分离线性/样条特征
    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    X_lin = X_obs[:, linear_cols] if linear_cols else None
    spline_bases = [_build_spline_basis(X_obs[:, c], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree']) for c in spline_cols]

    # 构建向量化week数据
    vec_week = _build_vectorized_week_data(
        train_data['week_data'],
        train_data['n_obs'],
        train_data['judge_score_pct'],
        train_data['judge_rank_score']
    )

    return {
        'train_data': train_data,
        'prior': config['prior'],
        'model_cfg': model_cfg,
        'X_lin': X_lin,
        'spline_bases': spline_bases,
        'vec_week': vec_week,
        'feature_means': feature_means,
        'X_obs_names': X_obs_names,
        'X_obs_names_orig': X_obs_names_orig,  # 保存原始特征名用于预测
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
             vec_week=model_params['vec_week'])

    print("    [train] MCMC completed")
    datas['mcmc'] = mcmc
    datas['trace'] = mcmc.get_samples()
    # 保存特征处理信息用于预测
    datas['feature_means'] = model_params.get('feature_means')
    datas['X_obs_names_filtered'] = model_params.get('X_obs_names')
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

    for i in range(20):
        key = f'spline_{i}_coef'
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
    week_data = datas['train_data']['week_data']

    n_samples = S_samples.shape[0]
    week_results = []

    for wd in week_data:
        k = wd['n_eliminated']
        if k == 0:
            week_results.append({'accuracy': None, 'season': wd['season'], 'week': wd['week']})
            continue

        mask = wd['obs_mask']
        indices = np.where(mask)[0]

        elim_global = np.where(wd['eliminated_mask'])[0]
        actual_elim = set(np.searchsorted(indices, elim_global))

        correct_count = 0
        for s in range(n_samples):
            S_week = S_samples[s, mask]

            if wd['judge_save_active']:
                pred_danger = set(np.argsort(S_week)[:2])
                correct = actual_elim.issubset(pred_danger)
            else:
                pred_elim = set(np.argsort(S_week)[:k])
                correct = pred_elim == actual_elim

            if correct:
                correct_count += 1

        accuracy = correct_count / n_samples
        week_results.append({
            'accuracy': accuracy,
            'season': wd['season'],
            'week': wd['week'],
            'judge_save': wd['judge_save_active'],
        })

    valid_acc = [r['accuracy'] for r in week_results if r['accuracy'] is not None]
    datas['metrics'] = {
        'week_results': week_results,
        'mean_accuracy': np.mean(valid_acc) if valid_acc else 0.0,
        'n_weeks_evaluated': len(valid_acc),
    }

    print(f"Mean accuracy: {datas['metrics']['mean_accuracy']:.3f} ({len(valid_acc)} weeks)")
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

    # 获取原始特征
    X_obs_orig = td['X_obs'].copy()
    X_obs_names_orig = list(td['X_obs_names'])

    X_obs = X_obs_orig.copy()
    X_obs_names = list(X_obs_names_orig)
    spline_features = model_cfg['spline_features'] or []
    exclude_features = model_cfg.get('exclude_features') or []
    interaction_features = model_cfg.get('interaction_features') or []
    center_features = model_cfg.get('center_features', False)

    # 添加交互项
    if interaction_features:
        for expr in interaction_features:
            new_col, new_name = _parse_interaction(expr, X_obs_orig, X_obs_names_orig)
            X_obs = np.column_stack([X_obs, new_col])
            X_obs_names.append(new_name)

    # 过滤掉排除的特征
    if exclude_features:
        keep_cols = [i for i, name in enumerate(X_obs_names) if name not in exclude_features]
        X_obs = X_obs[:, keep_cols]
        X_obs_names = [X_obs_names[i] for i in keep_cols]

    # 使用训练集的均值进行中心化
    feature_means = datas.get('feature_means')
    if center_features and feature_means is not None:
        X_obs = X_obs - feature_means

    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]
    linear_names = [X_obs_names[i] for i in linear_cols]
    spline_names = [X_obs_names[i] for i in spline_cols]

    X_lin = X_obs[:, linear_cols] if linear_cols else None

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
    for i, col in enumerate(spline_cols):
        key = f'spline_{i}_coef'
        if key in posterior:
            coef = posterior[key].mean(axis=0)
            spline_coefs.append(coef)
            basis = _build_spline_basis(X_obs[:, col], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree'])
            contrib_i = basis @ coef
            spline_contrib = spline_contrib + contrib_i
            spline_details[f'spline_{spline_names[i]}'] = contrib_i

    mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib

    P_fan = np.zeros(n_obs, dtype=np.float32)
    for wd in td['week_data']:
        mask = wd['obs_mask']
        P_fan[mask] = softmax(mu[mask])

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
