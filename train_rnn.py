"""RNN模型 - 捕捉时序特征看上限"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

from src.utils import load_config, load_data


class ContestantDataset(Dataset):
    """按选手组织的时序数据集"""
    def __init__(self, sequences, labels):
        self.sequences = sequences  # list of (seq_len, n_features)
        self.labels = labels        # list of (seq_len,)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def build_sequences(config, train_data):
    """构建按选手的时序数据"""
    n_obs = train_data['n_obs']
    n_celebs = train_data['n_celebs']
    week_data = train_data['week_data']

    # 获取所有特征
    X_obs = train_data['X_obs']
    X_celeb = train_data['X_celeb']
    celeb_idx = train_data['celeb_idx']
    obs_names = train_data['X_obs_names']
    celeb_names = train_data['X_celeb_names']

    # 构建淘汰标签
    eliminated = np.zeros(n_obs, dtype=np.float32)
    for wd in week_data:
        eliminated[wd['eliminated_mask']] = 1

    # 按选手分组
    celeb_to_obs = {c: [] for c in range(n_celebs)}
    for i in range(n_obs):
        celeb_to_obs[celeb_idx[i]].append(i)

    # 构建序列
    sequences = []
    labels = []
    celeb_ids = []

    for c in range(n_celebs):
        obs_indices = celeb_to_obs[c]
        if len(obs_indices) == 0:
            continue

        # 按周排序
        obs_indices = sorted(obs_indices, key=lambda i: train_data['week_idx'][i])

        # 特征：观测特征 + 名人特征
        seq_obs = X_obs[obs_indices]
        seq_celeb = np.tile(X_celeb[c], (len(obs_indices), 1))
        seq = np.hstack([seq_obs, seq_celeb])

        # 标签
        seq_labels = eliminated[obs_indices]

        sequences.append(torch.FloatTensor(seq))
        labels.append(torch.FloatTensor(seq_labels))
        celeb_ids.append(c)

    feature_names = obs_names + ['celeb_' + n for n in celeb_names]
    return sequences, labels, celeb_ids, feature_names


def collate_fn(batch):
    """处理变长序列"""
    sequences, labels = zip(*batch)
    lengths = [len(s) for s in sequences]
    max_len = max(lengths)
    n_features = sequences[0].shape[1]

    # Padding
    padded_seq = torch.zeros(len(sequences), max_len, n_features)
    padded_labels = torch.zeros(len(sequences), max_len)
    mask = torch.zeros(len(sequences), max_len)

    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        padded_seq[i, :len(seq)] = seq
        padded_labels[i, :len(lab)] = lab
        mask[i, :len(seq)] = 1

    return padded_seq, padded_labels, mask, lengths


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x, lengths=None):
        # x: (batch, seq_len, features)
        out, _ = self.rnn(x)
        # out: (batch, seq_len, hidden*2)
        out = self.fc(out).squeeze(-1)
        return out  # (batch, seq_len)


def evaluate_per_week(model, sequences, labels, celeb_ids, train_data, device):
    """按周评估淘汰预测准确率"""
    model.eval()
    week_data = train_data['week_data']
    celeb_idx = train_data['celeb_idx']

    # 获取所有预测分数
    all_scores = {}  # obs_idx -> score
    with torch.no_grad():
        for seq, lab, cid in zip(sequences, labels, celeb_ids):
            seq = seq.unsqueeze(0).to(device)
            scores = model(seq).squeeze(0).cpu().numpy()

            # 找到该选手的观测索引
            obs_indices = np.where(celeb_idx == cid)[0]
            obs_indices = sorted(obs_indices, key=lambda i: train_data['week_idx'][i])

            for i, obs_i in enumerate(obs_indices):
                all_scores[obs_i] = scores[i]

    # 按周评估
    correct = 0
    total = 0

    for w, wd in enumerate(week_data):
        if wd['n_eliminated'] == 0:
            continue

        obs_mask = wd['obs_mask']
        elim_mask = wd['eliminated_mask']
        n_elim = wd['n_eliminated']

        week_obs = np.where(obs_mask)[0]
        week_scores = np.array([all_scores.get(i, 0) for i in week_obs])

        # 预测：分数最高的 n_elim 个被淘汰（因为我们预测的是淘汰概率）
        pred_elim_idx = np.argsort(week_scores)[-n_elim:]

        # 真实淘汰者
        true_elim_global = np.where(elim_mask)[0]
        true_elim_idx = [np.where(week_obs == g)[0][0] for g in true_elim_global]

        correct += len(set(pred_elim_idx) & set(true_elim_idx))
        total += n_elim

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 60)
    print("RNN模型特征上限测试")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载数据
    config = load_config('config.yaml')
    datas = load_data('data/datas.pkl')
    train_data = datas['train_data']

    # 构建序列
    sequences, labels, celeb_ids, feature_names = build_sequences(config, train_data)

    print(f"\n数据规模:")
    print(f"  选手数: {len(sequences)}")
    print(f"  特征数: {sequences[0].shape[1]}")
    print(f"  平均序列长度: {np.mean([len(s) for s in sequences]):.1f}")

    # 标准化
    all_data = torch.cat(sequences, dim=0)
    mean = all_data.mean(dim=0)
    std = all_data.std(dim=0) + 1e-8
    sequences = [(s - mean) / std for s in sequences]

    # 创建数据集
    dataset = ContestantDataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # 模型
    input_size = sequences[0].shape[1]
    model = RNNModel(input_size, hidden_size=64, num_layers=2, dropout=0.3).to(device)

    # 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    print("\n训练中...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch_seq, batch_labels, batch_mask, lengths in dataloader:
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)
            batch_mask = batch_mask.to(device)

            optimizer.zero_grad()
            outputs = model(batch_seq)

            # 只计算有效位置的损失
            loss = criterion(outputs, batch_labels)
            loss = (loss * batch_mask).sum() / batch_mask.sum()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            acc = evaluate_per_week(model, sequences, labels, celeb_ids, train_data, device)
            print(f"  Epoch {epoch+1:3d}: Loss={total_loss/len(dataloader):.4f}, 准确率={acc:.3f}")

    # 最终评估
    print("\n" + "-" * 60)
    print("最终结果")
    print("-" * 60)
    acc = evaluate_per_week(model, sequences, labels, celeb_ids, train_data, device)
    print(f"  淘汰预测准确率: {acc:.3f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
