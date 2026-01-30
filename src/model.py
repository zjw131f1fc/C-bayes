"""贝叶斯模型"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from patsy import dmatrix


def _hierarchical_intercept(name, X, n_units, beta_scale, sigma_a, sigma_b, sum_to_zero=True):
    """分层随机截距: effect = β^T X + σ * ε"""
    n_features = X.shape[1]
    beta = pm.Normal(f'{name}_beta', mu=0, sigma=beta_scale, shape=n_features)
    sigma = pm.InverseGamma(f'{name}_sigma', alpha=sigma_a, beta=sigma_b)
    eps = pm.Normal(f'{name}_eps', mu=0, sigma=1, shape=n_units)

    effect_raw = pm.math.dot(X, beta) + sigma * eps
    if sum_to_zero:
        return pm.Deterministic(name, effect_raw - pm.math.mean(effect_raw))
    return pm.Deterministic(name, effect_raw)


def _build_spline_basis(x, n_knots, degree):
    """构建B样条基矩阵"""
    knots = np.linspace(x.min(), x.max(), n_knots)
    basis = dmatrix(
        f"bs(x, knots={list(knots[1:-1])}, degree={degree}, include_intercept=False) - 1",
        {"x": x}, return_type='dataframe'
    )
    return np.asarray(basis, dtype=np.float32)


def _linear_contrib(X_lin, prior):
    """线性特征贡献: β^T X_lin"""
    if X_lin is None or X_lin.shape[1] == 0:
        return 0
    n_linear = X_lin.shape[1]
    beta_obs = pm.Normal('beta_obs', mu=0, sigma=prior['beta_obs_scale'], shape=n_linear)
    return pm.math.dot(X_lin, beta_obs)


def _spline_contrib(spline_bases, prior):
    """样条特征贡献: Σ f_k(x_k)"""
    contrib = 0
    for i, basis in enumerate(spline_bases):
        n_basis = basis.shape[1]
        coef = pm.Normal(f'spline_{i}_coef', mu=0,
                         sigma=prior['spline_smoothing_scale'], shape=n_basis)
        contrib = contrib + pm.math.dot(basis, coef)
    return contrib


def _grouped_softmax(mu, week_idx, n_weeks):
    """按周分组的 Softmax: P_fan[i] = exp(mu[i]) / Σ_{j in same week} exp(mu[j])"""
    # 计算每周的 logsumexp
    week_logsumexp = pt.zeros(n_weeks)
    for w in range(n_weeks):
        mask = pt.eq(week_idx, w)
        # 用 -inf 屏蔽非当周选手
        mu_masked = pt.switch(mask, mu, -1e10)
        week_logsumexp = pt.set_subtensor(
            week_logsumexp[w],
            pm.math.logsumexp(mu_masked)
        )
    # P_fan = exp(mu - logsumexp[week_idx])
    log_P = mu - week_logsumexp[week_idx]
    return pm.math.exp(log_P)


def _soft_rank_score(P_fan, week_idx, n_weeks, week_data, tau=0.1):
    """
    软排名分: R'_fan = soft_rank / (n - 1)
    soft_rank[i] = Σ_j sigmoid((P_fan[i] - P_fan[j]) / τ)  (同周内)
    """
    n_obs = P_fan.shape[0]
    R_fan = pt.zeros(n_obs)

    for w in range(n_weeks):
        mask = week_data[w]['obs_mask']
        n_contestants = week_data[w]['n_contestants']
        if n_contestants <= 1:
            continue

        # 当周选手的 P_fan
        indices = np.where(mask)[0]
        P_w = P_fan[indices]  # [n_contestants]

        # 两两比较: diff[i,j] = P_w[i] - P_w[j]
        diff = P_w.dimshuffle(0, 'x') - P_w.dimshuffle('x', 0)  # [n, n]
        # soft_rank[i] = 有多少人比我低 (不含自己)
        soft_rank = pt.sum(pt.sigmoid(diff / tau), axis=1) - 0.5  # 减去自己
        # 归一化到 [0, 1]
        R_w = soft_rank / (n_contestants - 1)

        # 写回
        for idx_local, idx_global in enumerate(indices):
            R_fan = pt.set_subtensor(R_fan[idx_global], R_w[idx_local])

    return R_fan


def _compute_score(P_fan, judge_score_pct, judge_rank_score, week_idx, n_weeks, week_data, tau=0.1):
    """
    综合得分 S: 根据 rule_method 选择计算方式
    - 百分比法 (rule_method=1): S = judge_score_pct + P_fan
    - 排名法 (rule_method=0): S = judge_rank_score + soft_rank_score(P_fan)
    """
    n_obs = P_fan.shape[0]

    # 先计算软排名分（排名法需要）
    R_fan = _soft_rank_score(P_fan, week_idx, n_weeks, week_data, tau)

    # 根据每周的 rule_method 选择
    S = pt.zeros(n_obs)
    for w in range(n_weeks):
        mask = week_data[w]['obs_mask']
        indices = np.where(mask)[0]
        rule = week_data[w]['rule_method']

        for idx in indices:
            if rule == 1:  # 百分比法
                s_val = judge_score_pct[idx] + P_fan[idx]
            else:  # 排名法
                s_val = judge_rank_score[idx] + R_fan[idx]
            S = pt.set_subtensor(S[idx], s_val)

    return S


def _plackett_luce_loglik(S, week_data, tau):
    """
    Plackett-Luce 对数似然: 淘汰者应该有最低的 S
    L_w = Π_{e in eliminated} exp(-τ*S_e) / Σ_{j in remaining} exp(-τ*S_j)
    """
    total_ll = 0

    for w, wd in enumerate(week_data):
        n_elim = wd['n_eliminated']
        if n_elim == 0:
            continue

        # 当周选手索引
        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        indices = np.where(obs_mask)[0]
        elim_indices = np.where(elim_mask)[0]

        # 当周选手的 S
        S_w = S[indices]
        n_contestants = len(indices)

        # 淘汰者在当周的局部索引
        elim_local = [np.where(indices == e)[0][0] for e in elim_indices]

        # 初始化剩余选手掩码
        remaining = pt.ones(n_contestants)

        # 顺序选出淘汰者
        for e_local in elim_local:
            # 计算 logsumexp(-τ*S) for remaining
            neg_tau_S = -tau * S_w
            log_denom = pm.math.logsumexp(neg_tau_S + pt.log(remaining + 1e-10))

            # 当前淘汰者的对数似然贡献
            ll = neg_tau_S[e_local] - log_denom
            total_ll = total_ll + ll

            # 从剩余池中移除
            remaining = pt.set_subtensor(remaining[e_local], 0.0)

    return total_ll


def build_model(config, datas):
    """构建PyMC分层贝叶斯模型"""
    train_data = datas['train_data']
    prior = config['prior']
    model_cfg = config['model']

    # 索引
    celeb_idx = train_data['celeb_idx']
    pro_idx = train_data['pro_idx']
    week_idx = train_data['week_idx']
    n_weeks = train_data['n_weeks']

    # 分离线性/样条特征
    X_obs = train_data['X_obs']
    X_obs_names = train_data['X_obs_names']
    spline_features = model_cfg['spline_features']

    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    X_lin = X_obs[:, linear_cols] if linear_cols else None
    spline_bases = [_build_spline_basis(X_obs[:, c], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree']) for c in spline_cols]

    with pm.Model() as model:
        # 名人效应 α: [n_celebs]
        alpha = _hierarchical_intercept(
            'alpha', train_data['X_celeb'], train_data['n_celebs'],
            prior['beta_celeb_scale'],
            prior['sigma_celeb']['a'], prior['sigma_celeb']['b']
        )

        # 舞者效应 δ: [n_pros]
        delta = _hierarchical_intercept(
            'delta', train_data['X_pro'], train_data['n_pros'],
            prior['beta_pro_scale'],
            prior['sigma_pro']['a'], prior['sigma_pro']['b']
        )

        # 观测级特征
        lin_contrib = _linear_contrib(X_lin, prior)
        spline_contrib = _spline_contrib(spline_bases, prior)

        # 观测级异方差: σ_obs = σ_base * exp(β_σ^T X_obs)
        if model_cfg.get('use_heteroscedastic', False):
            n_obs = train_data['n_obs']
            sigma_base = pm.InverseGamma('sigma_base', alpha=prior['sigma_obs']['a'],
                                         beta=prior['sigma_obs']['b'])
            beta_sigma = pm.Normal('beta_sigma', mu=0, sigma=prior.get('beta_sigma_scale', 0.5),
                                   shape=train_data['X_obs'].shape[1])
            log_mult = pm.math.dot(train_data['X_obs'], beta_sigma)
            sigma_obs = sigma_base * pm.math.exp(log_mult)
            eps_obs = pm.Normal('eps_obs', mu=0, sigma=1, shape=n_obs)
            obs_noise = sigma_obs * eps_obs
        else:
            obs_noise = 0

        # 潜在投票强度 μ: [n_obs]
        mu = alpha[celeb_idx] + delta[pro_idx] + lin_contrib + spline_contrib + obs_noise
        mu = pm.Deterministic('mu', mu)

        # 粉丝票占比 P_fan: [n_obs]
        P_fan = _grouped_softmax(mu, week_idx, n_weeks)
        P_fan = pm.Deterministic('P_fan', P_fan)

        # 综合得分 S: [n_obs]
        S = _compute_score(
            P_fan,
            train_data['judge_score_pct'],
            train_data['judge_rank_score'],
            week_idx, n_weeks,
            train_data['week_data']
        )
        S = pm.Deterministic('S', S)

        # 似然函数温度参数 τ
        if model_cfg['learn_tau']:
            tau = pm.HalfNormal('tau', sigma=model_cfg['tau_init'])
        else:
            tau = model_cfg['tau_init']

        # Plackett-Luce 似然
        log_lik = _plackett_luce_loglik(S, train_data['week_data'], tau)
        pm.Potential('likelihood', log_lik)

    return model


def train(config, model, datas):
    """MCMC采样，trace 存入 datas['trace']"""
    import sys
    samp = config['sampling']

    print(f"    [train] Starting MCMC: {samp['n_chains']} chains, {samp['n_tune']} tune, {samp['n_samples']} samples")
    sys.stdout.flush()

    try:
        with model:
            trace = pm.sample(
                draws=samp['n_samples'],
                tune=samp['n_tune'],
                chains=samp['n_chains'],
                target_accept=samp['target_accept'],
                random_seed=samp['random_seed'],
                cores=samp['cores'],
                return_inferencedata=True,
                progressbar=True
            )
        print("    [train] MCMC completed")
    except Exception as e:
        print(f"    [train] ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise

    datas['trace'] = trace
    return datas


def extract_posterior(config, datas):
    """从 trace 提取后验样本"""
    trace = datas['trace']
    posterior = trace.posterior

    # 合并 chain 和 draw 维度: [n_chains, n_draws, ...] -> [n_samples, ...]
    def flatten_samples(var):
        shape = var.shape
        return var.values.reshape(-1, *shape[2:])

    samples = {
        'alpha': flatten_samples(posterior['alpha']),
        'delta': flatten_samples(posterior['delta']),
    }

    # beta_obs（如果存在）
    if 'beta_obs' in posterior:
        samples['beta_obs'] = flatten_samples(posterior['beta_obs'])

    # spline 系数
    for i in range(20):
        key = f'spline_{i}_coef'
        if key in posterior:
            samples[key] = flatten_samples(posterior[key])
        else:
            break

    # tau
    if 'tau' in posterior:
        samples['tau'] = posterior['tau'].values.flatten()

    datas['posterior_samples'] = samples
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

        # 实际淘汰者（局部索引）
        elim_global = np.where(wd['eliminated_mask'])[0]
        actual_elim = set(np.searchsorted(indices, elim_global))

        correct_count = 0
        for s in range(n_samples):
            S_week = S_samples[s, mask]

            if wd['judge_save_active']:
                # 评委拯救：预测进入危险区的 2 人
                pred_danger = set(np.argsort(S_week)[:2])
                correct = actual_elim.issubset(pred_danger)
            else:
                # 正常：预测 S 最低的 k 个
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

    # 汇总
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

    # 后验均值
    alpha = posterior['alpha'].mean(axis=0)  # [n_celebs]
    delta = posterior['delta'].mean(axis=0)  # [n_pros]

    # 索引
    celeb_idx = td['celeb_idx']
    pro_idx = td['pro_idx']
    week_idx = td['week_idx']
    n_obs = td['n_obs']

    # 分离线性/样条特征
    X_obs = td['X_obs']
    X_obs_names = td['X_obs_names']
    spline_features = model_cfg['spline_features']

    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]
    linear_names = [X_obs_names[i] for i in linear_cols]
    spline_names = [X_obs_names[i] for i in spline_cols]

    X_lin = X_obs[:, linear_cols] if linear_cols else None

    # 贡献分解
    alpha_contrib = alpha[celeb_idx]
    delta_contrib = delta[pro_idx]

    # 线性贡献
    linear_contrib = np.zeros(n_obs, dtype=np.float32)
    linear_details = {}
    if X_lin is not None and 'beta_obs' in posterior:
        beta_obs = posterior['beta_obs'].mean(axis=0)
        linear_contrib = X_lin @ beta_obs
        # 细分
        for i, name in enumerate(linear_names):
            linear_details[f'linear_{name}'] = X_lin[:, i] * beta_obs[i]

    # 样条贡献
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

    # mu
    mu = alpha_contrib + delta_contrib + linear_contrib + spline_contrib

    # P_fan (周内 softmax)
    P_fan = np.zeros(n_obs, dtype=np.float32)
    for wd in td['week_data']:
        mask = wd['obs_mask']
        P_fan[mask] = softmax(mu[mask])

    # 构建输出
    output = {
        # 维度信息
        'n_obs': n_obs,
        'n_celebs': td['n_celebs'],
        'n_pros': td['n_pros'],
        # 观测索引
        'celeb_idx': celeb_idx,
        'pro_idx': pro_idx,
        'week_idx': week_idx,
        # 观测级预测
        'mu': mu,
        'P_fan': P_fan,
        # 观测级分解
        'alpha_contrib': alpha_contrib,
        'delta_contrib': delta_contrib,
        'linear_contrib': linear_contrib,
        'spline_contrib': spline_contrib,
        # 模型参数
        'alpha': alpha,
        'delta': delta,
        'beta_obs': posterior.get('beta_obs', np.array([])).mean(axis=0) if 'beta_obs' in posterior else np.array([]),
        'spline_coefs': spline_coefs,
        # 元信息
        'feature_names': {
            'beta_obs': linear_names,
            'splines': spline_names,
        },
    }

    # 添加细分贡献
    output.update(linear_details)
    output.update(spline_details)

    datas['model_output'] = output
    return datas
