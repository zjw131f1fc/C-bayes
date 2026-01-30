"""贝叶斯模型 (NumPyro 版本)"""

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


def _model_fn(train_data, prior, model_cfg, X_lin, spline_bases):
    """NumPyro 模型定义"""
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

    # 按周分组 softmax 计算 P_fan
    week_data = train_data['week_data']
    P_fan = jnp.zeros(n_obs)
    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        mu_week = mu[indices]
        P_week = jax.nn.softmax(mu_week)
        P_fan = P_fan.at[indices].set(P_week)
    numpyro.deterministic('P_fan', P_fan)

    # 综合得分 S
    S = jnp.zeros(n_obs)
    for wd in week_data:
        mask = wd['obs_mask']
        indices = np.where(mask)[0]
        rule = wd['rule_method']

        if rule == 1:  # 百分比法
            S_week = train_data['judge_score_pct'][indices] + P_fan[indices]
        else:  # 排名法
            P_week = P_fan[indices]
            n_contestants = len(indices)
            if n_contestants > 1:
                diff = P_week[:, None] - P_week[None, :]
                soft_rank = jnp.sum(jax.nn.sigmoid(diff / 0.1), axis=1) - 0.5
                R_fan = soft_rank / (n_contestants - 1)
            else:
                R_fan = jnp.array([0.5])
            S_week = train_data['judge_rank_score'][indices] + R_fan
        S = S.at[indices].set(S_week)
    numpyro.deterministic('S', S)

    # 似然函数温度参数 τ
    if model_cfg['learn_tau']:
        tau = numpyro.sample('tau', dist.HalfNormal(model_cfg['tau_init']))
    else:
        tau = model_cfg['tau_init']

    # Plackett-Luce 似然
    total_ll = 0.0
    for wd in week_data:
        n_elim = wd['n_eliminated']
        if n_elim == 0:
            continue

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        indices = np.where(obs_mask)[0]
        elim_indices = np.where(elim_mask)[0]

        S_w = S[indices]
        n_contestants = len(indices)
        elim_local = [np.where(indices == e)[0][0] for e in elim_indices]

        remaining = jnp.ones(n_contestants)
        for e_local in elim_local:
            neg_tau_S = -tau * S_w
            log_denom = jax.scipy.special.logsumexp(neg_tau_S + jnp.log(remaining + 1e-10))
            ll = neg_tau_S[e_local] - log_denom
            total_ll = total_ll + ll
            remaining = remaining.at[e_local].set(0.0)

    numpyro.factor('likelihood', total_ll)


def build_model(config, datas):
    """构建模型参数（NumPyro 不需要预构建模型对象）"""
    train_data = datas['train_data']
    model_cfg = config['model']

    # 分离线性/样条特征
    X_obs = train_data['X_obs']
    X_obs_names = train_data['X_obs_names']
    spline_features = model_cfg['spline_features']

    spline_cols = [i for i, name in enumerate(X_obs_names) if name in spline_features]
    linear_cols = [i for i, name in enumerate(X_obs_names) if name not in spline_features]

    X_lin = X_obs[:, linear_cols] if linear_cols else None
    spline_bases = [_build_spline_basis(X_obs[:, c], model_cfg['n_spline_knots'],
                                        model_cfg['spline_degree']) for c in spline_cols]

    # 返回模型配置（NumPyro 在 train 时才真正构建）
    return {
        'train_data': train_data,
        'prior': config['prior'],
        'model_cfg': model_cfg,
        'X_lin': X_lin,
        'spline_bases': spline_bases,
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
             spline_bases=model_params['spline_bases'])

    print("    [train] MCMC completed")
    datas['mcmc'] = mcmc
    datas['trace'] = mcmc.get_samples()
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

    X_obs = td['X_obs']
    X_obs_names = td['X_obs_names']
    spline_features = model_cfg['spline_features']

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
