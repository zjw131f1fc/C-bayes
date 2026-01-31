"""简单神经网络 - 用JAX/Flax实现，看特征上限"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, load_data


def build_features(config, train_data):
    """构建全特征矩阵"""
    X_obs = train_data['X_obs']
    X_celeb = train_data['X_celeb']
    celeb_idx = train_data['celeb_idx']

    # 扩展名人特征到观测级
    X_celeb_expanded = X_celeb[celeb_idx]

    # 合并
    X = np.hstack([X_obs, X_celeb_expanded])

    # 标准化
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    X = (X - mean) / std

    return X.astype(np.float32)


def build_labels(train_data):
    """构建标签"""
    week_data = train_data['week_data']
    n_obs = train_data['n_obs']

    y = np.ones(n_obs, dtype=np.float32)
    for wd in week_data:
        y[wd['eliminated_mask']] = 0

    return y


def init_mlp(key, input_size, hidden_sizes=[128, 64]):
    """初始化MLP参数"""
    params = {}
    sizes = [input_size] + hidden_sizes + [1]

    for i in range(len(sizes) - 1):
        key, subkey = random.split(key)
        scale = np.sqrt(2.0 / sizes[i])
        params[f'W{i}'] = random.normal(subkey, (sizes[i], sizes[i+1])) * scale
        params[f'b{i}'] = jnp.zeros(sizes[i+1])

    return params


def mlp_forward(params, x, training=True):
    """MLP前向传播"""
    n_layers = len([k for k in params.keys() if k.startswith('W')])

    for i in range(n_layers - 1):
        x = jnp.dot(x, params[f'W{i}']) + params[f'b{i}']
        x = jax.nn.relu(x)

    # 最后一层
    x = jnp.dot(x, params[f'W{n_layers-1}']) + params[f'b{n_layers-1}']
    return x.squeeze(-1)


def loss_fn(params, X, y):
    """BCE损失"""
    logits = mlp_forward(params, X)
    return optax.sigmoid_binary_cross_entropy(logits, y).mean()


def train_step(params, opt_state, X, y, optimizer):
    """训练步骤"""
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def evaluate_per_week(params, X, train_data):
    """按周评估"""
    week_data = train_data['week_data']
    scores = np.array(mlp_forward(params, X))

    correct = 0
    total = 0

    for wd in week_data:
        if wd['n_eliminated'] == 0:
            continue

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']

        week_obs = np.where(obs_mask)[0]
        week_scores = scores[obs_mask]

        # 分数最低的被淘汰
        pred_elim_idx = np.argsort(week_scores)[:n_elim]

        true_elim_global = np.where(elim_mask)[0]
        true_elim_idx = [np.where(week_obs == g)[0][0] for g in true_elim_global]

        correct += len(set(pred_elim_idx) & set(true_elim_idx))
        total += n_elim

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("MLP模型特征上限测试 (JAX) - 交叉验证")
    print("=" * 60)

    # 加载数据
    config = load_config('config.yaml')
    datas = load_data('data/datas.pkl')
    train_data = datas['train_data']

    # 构建特征
    X = build_features(config, train_data)
    y = build_labels(train_data)

    # 获取赛季信息
    season_idx = train_data['season_idx']
    unique_seasons = np.unique(season_idx)

    print(f"\n数据规模:")
    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  赛季数: {len(unique_seasons)}")

    # 按赛季交叉验证
    print("\n按赛季交叉验证...")
    all_correct = 0
    all_total = 0

    for test_season in unique_seasons:
        # 划分训练/测试
        train_mask = season_idx != test_season
        test_mask = season_idx == test_season

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        if X_test.shape[0] == 0:
            continue

        # 标准化（用训练集统计量）
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        # 转为JAX数组
        X_train_jax = jnp.array(X_train_norm)
        y_train_jax = jnp.array(y_train)
        X_test_jax = jnp.array(X_test_norm)

        # 初始化
        key = random.PRNGKey(42)
        params = init_mlp(key, X.shape[1], hidden_sizes=[64, 32])

        # 优化器
        optimizer = optax.adamw(learning_rate=0.001, weight_decay=1e-3)
        opt_state = optimizer.init(params)

        # 训练
        n_epochs = 100
        batch_size = 128

        for epoch in range(n_epochs):
            perm = np.random.permutation(len(X_train_norm))
            for i in range(0, len(X_train_norm), batch_size):
                idx = perm[i:i+batch_size]
                X_batch = X_train_jax[idx]
                y_batch = y_train_jax[idx]
                params, opt_state, _ = train_step(params, opt_state, X_batch, y_batch, optimizer)

        # 在测试集上评估
        week_data = train_data['week_data']
        scores = np.array(mlp_forward(params, X_test_jax))

        # 构建测试集的week_data
        test_obs_indices = np.where(test_mask)[0]
        correct = 0
        total = 0

        for wd in week_data:
            if wd['n_eliminated'] == 0:
                continue

            # 该周在测试集中的观测
            week_obs_global = np.where(wd['obs_mask'])[0]
            week_in_test = [i for i in week_obs_global if i in test_obs_indices]

            if len(week_in_test) == 0:
                continue

            n_elim = wd['n_eliminated']

            # 获取分数
            week_scores = []
            for obs_i in week_in_test:
                test_local_idx = np.where(test_obs_indices == obs_i)[0][0]
                week_scores.append(scores[test_local_idx])
            week_scores = np.array(week_scores)

            # 预测
            pred_elim_idx = np.argsort(week_scores)[:n_elim]

            # 真实淘汰
            true_elim_global = np.where(wd['eliminated_mask'])[0]
            true_elim_idx = [i for i, obs_i in enumerate(week_in_test) if obs_i in true_elim_global]

            correct += len(set(pred_elim_idx) & set(true_elim_idx))
            total += len(true_elim_idx)

        if total > 0:
            acc = correct / total
            print(f"  Season {test_season:2d}: {correct}/{total} = {acc:.3f}")
            all_correct += correct
            all_total += total

    # 总体结果
    print("\n" + "-" * 60)
    print("最终结果 (交叉验证)")
    print("-" * 60)
    overall_acc = all_correct / all_total if all_total > 0 else 0
    print(f"  淘汰预测准确率: {overall_acc:.3f} ({all_correct}/{all_total})")

    print("\n对比:")
    print(f"  线性模型:   56.0%")
    print(f"  MLP (CV):   {overall_acc*100:.1f}%")
    print(f"  贝叶斯模型: ~59.8%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
