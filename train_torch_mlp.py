"""MLP模型 - PyTorch实现，按赛季交叉验证"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, load_data


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_features(train_data):
    """构建全特征矩阵"""
    X_obs = train_data['X_obs']
    X_celeb = train_data['X_celeb']
    celeb_idx = train_data['celeb_idx']

    # 扩展名人特征到观测级
    X_celeb_expanded = X_celeb[celeb_idx]
    X = np.hstack([X_obs, X_celeb_expanded])

    return X.astype(np.float32)


def build_labels(train_data):
    """构建标签：未淘汰=1，被淘汰=0"""
    week_data = train_data['week_data']
    n_obs = train_data['n_obs']

    y = np.ones(n_obs, dtype=np.float32)
    for wd in week_data:
        y[wd['eliminated_mask']] = 0

    return y


def evaluate_per_week(model, X, train_data, test_mask, device):
    """按周评估淘汰预测准确率"""
    model.eval()
    with torch.no_grad():
        scores = model(torch.FloatTensor(X).to(device)).cpu().numpy()

    week_data = train_data['week_data']
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
        week_scores = scores[week_in_test]

        # 预测：分数最低的被淘汰
        pred_elim_idx = np.argsort(week_scores)[:n_elim]

        # 真实淘汰
        true_elim_global = np.where(wd['eliminated_mask'])[0]
        true_elim_idx = [i for i, obs_i in enumerate(week_in_test) if obs_i in true_elim_global]

        correct += len(set(pred_elim_idx) & set(true_elim_idx))
        total += len(true_elim_idx)

    return correct / total if total > 0 else 0.0, correct, total


def train_model(model, X_train, y_train, device, epochs=100, batch_size=128, lr=0.001):
    """训练模型"""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    return model


def main():
    print("=" * 60)
    print("MLP模型 (PyTorch) - 按赛季交叉验证")
    print("=" * 60)

    device = torch.device('cuda')  # 强制用CPU
    print(f"Device: {device}")

    # 加载数据
    config = load_config('config.yaml')
    datas = load_data('data/datas.pkl')
    train_data = datas['train_data']

    # 构建特征
    X = build_features(train_data)
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

        if X[test_mask].shape[0] == 0:
            continue

        # 标准化（用训练集统计量）
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mean) / std
        X_norm = (X - mean) / std  # 全部数据标准化，用于评估

        # 初始化模型
        model = MLP(X.shape[1], hidden_sizes=[64, 32], dropout=0.3).to(device)

        # 训练
        model = train_model(model, X_train_norm, y_train, device, epochs=100)

        # 评估
        acc, correct, total = evaluate_per_week(model, X_norm, train_data, test_mask, device)

        if total > 0:
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
    print(f"  线性模型 (全特征): 56.0%")
    print(f"  MLP (CV):          {overall_acc*100:.1f}%")
    print(f"  贝叶斯模型:        ~59.8%")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
