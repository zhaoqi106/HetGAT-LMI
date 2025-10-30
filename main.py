import torch
import pandas as pd
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, HeteroConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)

# ========== 1. 设置随机种子 ==========
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)

# ========== 2. 加载图数据 ==========
graph_path = 'D:\pytorchProject\RR\dateset\cfhan_hetero_graph.pt'
data = torch.load(graph_path)

# ========== 3. 加载交互数据并构造正负样本池 ==========
interaction_data = pd.read_csv('D:\pytorchProject\RR\dateset\mirna_lncrna_interaction.csv')
lnc_set = interaction_data['lncrna'].unique().tolist()
mi_set = interaction_data['mirna'].unique().tolist()
lnc_map = {n: i for i, n in enumerate(lnc_set)}
mi_map = {n: i for i, n in enumerate(mi_set)}
interaction_data['lnc_idx'] = interaction_data['lncrna'].map(lnc_map)
interaction_data['mi_idx'] = interaction_data['mirna'].map(mi_map)

positive_pairs = interaction_data[['lnc_idx', 'mi_idx']].values

# 构造负样本全集（不与正样本重合）
lnc_n = len(lnc_set)
mi_n = len(mi_set)
all_pairs = {(i, j) for i in range(lnc_n) for j in range(mi_n)}
positive_set = set(map(tuple, positive_pairs))
negative_pairs = np.array(list(all_pairs - positive_set), dtype=int)

# 固定 20% 负样本为 hold-out 验证池
holdout_frac = 0.2
num_hold = int(len(negative_pairs) * holdout_frac)
rng0 = np.random.RandomState(42)
hold_idx = rng0.choice(len(negative_pairs), num_hold, replace=False)
holdout_neg = negative_pairs[hold_idx]
train_neg_cands = np.delete(negative_pairs, hold_idx, axis=0)


# ========== 4. 模型结构（含门控融合 + 正则化改进） ==========

class GateLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),  # 修改：使用ReLU替代BatchNorm，避免维度问题
            nn.Sigmoid()
        )

    def forward(self, f1, f2):
        g = self.gate(torch.cat([f1, f2], dim=1))
        return g * f1 + (1 - g) * f2


class HetGNNLayer(nn.Module):
    def __init__(self, in_ch, out_dim, heads=1):
        super().__init__()
        # 修改：使用GATv2Conv替代GATConv
        self.conv = HeteroConv({
            ('lnc', 'similar', 'lnc'): GATv2Conv(in_ch['lnc'], out_dim, heads=heads, add_self_loops=False),
            ('mi', 'similar', 'mi'): GATv2Conv(in_ch['mi'], out_dim, heads=heads, add_self_loops=False),
            ('lnc', 'interacts', 'mi'): GATv2Conv((in_ch['lnc'], in_ch['mi']), out_dim, heads=heads,
                                                add_self_loops=False),
        }, aggr='mean')

    def forward(self, x_dict, edge_index_dict):
        return self.conv(x_dict, edge_index_dict)


class HetGNN(nn.Module):
    def __init__(self, in_ch, hid, heads=2):
        super().__init__()
        self.encoder = HetGNNLayer(in_ch, hid, heads)

        # 修改：简化特征融合，移除有问题的多头注意力
        feature_dim = hid * heads
        self.gate_layer = GateLayer(feature_dim)
        self.dropout_gate = nn.Dropout(0.3)  # 修改：降低dropout率

        # 修改：优化分类器结构
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, x_dict, edge_index_dict, pairs):
        # 编码节点特征
        x = self.encoder(x_dict, edge_index_dict)
        h_l, h_m = x['lnc'], x['mi']

        # 修改：正确提取节点对的特征
        src, tgt = pairs[:, 0], pairs[:, 1]
        lnc_features = h_l[src]  # 获取lncRNA特征
        mi_features = h_m[tgt]  # 获取miRNA特征

        # 门控融合
        fused = self.gate_layer(lnc_features, mi_features)
        fused = self.dropout_gate(fused)

        return self.classifier(fused).squeeze(1)


# ========== 5. 评估函数 ==========
def evaluate(pred, label):
    p = pred.detach().cpu().numpy()
    y = label.detach().cpu().numpy().astype(int)
    p_bin = (p > 0.5).astype(int)
    return {
        'AUC': roc_auc_score(y, p),
        'AUPR': average_precision_score(y, p),
        'Precision': precision_score(y, p_bin, zero_division=0),
        'Recall': recall_score(y, p_bin, zero_division=0),
        'F1': f1_score(y, p_bin, zero_division=0),
        'ACC': accuracy_score(y, p_bin)
    }


# ========== 6. 修复后的训练函数 ==========
def train_and_evaluate(model, data,
                       tr_pairs, tr_labels,
                       vl_pairs, vl_labels,
                       epochs=2000, lr=1e-2,
                       run_id=None, log_interval=50,
                       early_stop_patience=15):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = model.to(device), data.to(device)
    tr_pairs, tr_labels = tr_pairs.to(device), tr_labels.to(device)
    vl_pairs, vl_labels = vl_pairs.to(device), vl_labels.to(device)

    # 修改：调整优化器参数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 修改：调整学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5  # 提高最小学习率
    )

    best_auc, best_state, no_improve = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        # 修改：直接使用模型输出计算loss，不需要额外的sigmoid
        logits = model(data.x_dict, data.edge_index_dict, tr_pairs)
        loss = F.binary_cross_entropy_with_logits(logits, tr_labels.float())

        loss.backward()

        # 修改：添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        if epoch % log_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                tr_logits = model(data.x_dict, data.edge_index_dict, tr_pairs)
                vl_logits = model(data.x_dict, data.edge_index_dict, vl_pairs)

                tr_m = evaluate(torch.sigmoid(tr_logits), tr_labels)
                vl_m = evaluate(torch.sigmoid(vl_logits), vl_labels)

            # 修改：添加loss打印用于调试
            print(
                f"[Run {run_id}] Epoch {epoch} | Loss={loss.item():.4f} | Train AUC={tr_m['AUC']:.4f}, Val AUC={vl_m['AUC']:.4f}, F1={vl_m['F1']:.4f}")

            if vl_m['AUC'] > best_auc:
                best_auc = vl_m['AUC']
                best_state = model.state_dict().copy()
                no_improve = 0
                print(f"[Run {run_id}] 🌟 New best AUC={best_auc:.4f}")
            else:
                no_improve += 1
                if no_improve >= early_stop_patience:
                    print(f"[Run {run_id}] ⏹ Early stop at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vl_logits = model(data.x_dict, data.edge_index_dict, vl_pairs)
        final = evaluate(torch.sigmoid(vl_logits), vl_labels)
    print(f"[Run {run_id}] 🎯 Final Val AUC={final['AUC']:.4f}\n")
    return final

in_channels_dict = {
    'lnc': data['lnc'].x.size(1),
    'mi': data['mi'].x.size(1)
}

n_runs =50
results = []


# 模型保存函数
def save_model(model, run_id, file_path='model_run_{}.pth'):
    torch.save(model.state_dict(), file_path.format(run_id))
    print(f"Model for run {run_id} saved at {file_path.format(run_id)}")


for run in range(1, n_runs + 1):
    set_seed(42 + run)
    rng = np.random.RandomState(42 + run)

    # 正样本划分
    pos_train, pos_val = train_test_split(positive_pairs, test_size=0.2, random_state=42 + run)

    # 训练负样本
    neg_train_idx = rng.choice(len(train_neg_cands), size=len(pos_train), replace=False)
    neg_train = train_neg_cands[neg_train_idx]

    # 验证负样本（固定池）
    neg_val_idx = rng.choice(len(holdout_neg), size=len(pos_val), replace=False)
    neg_val = holdout_neg[neg_val_idx]

    # 构建训练集
    tr_pairs = torch.tensor(np.vstack([pos_train, neg_train]), dtype=torch.long)
    tr_labels = torch.cat([torch.ones(len(pos_train)), torch.zeros(len(neg_train))])

    vl_pairs = torch.tensor(np.vstack([pos_val, neg_val]), dtype=torch.long)
    vl_labels = torch.cat([torch.ones(len(pos_val)), torch.zeros(len(neg_val))])

    print(f"\n=== Run {run}/{n_runs} ===")

    # 修改：调整模型参数
    model = HetGNN(in_channels_dict, 128, heads=8)

    # 训练并评估模型
    result = train_and_evaluate(model, data, tr_pairs, tr_labels, vl_pairs, vl_labels,
                                run_id=run, lr=1e-4)  # 提高初始学习率

    # 保存模型
    save_model(model, run)  # 每轮训练后保存模型

    results.append(result)

# ========== 8. 汇总结果 ==========

keys = results[0].keys()
mean = {k: np.mean([r[k] for r in results]) for k in keys}
std = {k: np.std([r[k] for r in results]) for k in keys}
print("\n===== Summary over all runs =====")
for k in sorted(keys):
    print(f"{k:>9}: {mean[k]:.4f} ± {std[k]:.4f}")