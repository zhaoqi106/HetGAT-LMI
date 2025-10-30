# build_graph_autofix_v2.py
import os, numpy as np, pandas as pd, torch
from torch_geometric.data import HeteroData

# ===== 路径 =====
BASE = r"D:/pytorchProject/RR/dateset"
LNC_FEAT = os.path.join(BASE, "lncrna_final_features.csv")
MI_FEAT  = os.path.join(BASE, "mirna_final_features.csv")
TRAIN_TXT = os.path.join(BASE, "train.txt")
LNC_KNN = os.path.join(BASE, "lnc_knn_edges.csv")
MI_KNN  = os.path.join(BASE, "mi_knn_edges.csv")
OUT_PT  = os.path.join(BASE, "cfhan_hetero_graph.pt")

# ===== I/O =====
def load_array_anysep(path, cols=2, dtype=int):
    try: arr = np.loadtxt(path, dtype=dtype, delimiter=",")
    except Exception: arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim == 1: arr = arr.reshape(-1, cols)
    assert arr.shape[1] == cols, f"{path} 需要 {cols} 列，得到 {arr.shape[1]}"
    return torch.tensor(arr.T, dtype=torch.long)

def load_edges_csv(path):
    df = pd.read_csv(path, header=None).dropna().astype(int)
    assert df.shape[1] == 2, f"{path} 需要两列，得到 {df.shape[1]}"
    return torch.tensor(df.to_numpy().T, dtype=torch.long)

def symm(ei): return torch.cat([ei, ei.flip(0)], dim=1)

# ===== 规范化 =====
def ok(ei, Ln, Mn):
    return ei[0].min().item() >= 0 and ei[1].min().item() >= 0 and \
           ei[0].max().item() < Ln and ei[1].max().item() < Mn

def normalize_inter_edges(ei, Ln, Mn):
    """
    返回满足：列0是 lnc∈[0,Ln-1]，列1是 mi∈[0,Mn-1]
    尝试：原始/交换；各自再做 1-based 校正；再尝试 mi 列减去 Ln 的全局偏移。
    """
    cand_list = []
    # 原始与交换
    for base in [ei, ei.flip(0)]:
        cand_list.append(base)
        cand_list.append(base - 1)  # 1-based
        # mi 列可能是全局偏移（如 Ln..Ln+Mn-1）
        offset = base.clone()
        offset[1] = offset[1] - Ln
        cand_list.append(offset)
        cand_list.append(offset - 1)  # 同时 1-based

    for cand in cand_list:
        if ok(cand, Ln, Mn):
            return cand

    # 失败时给出诊断
    def rng(x): return int(x.min().item()), int(x.max().item())
    a0, a1 = rng(ei[0]), rng(ei[1])
    b0, b1 = rng(ei.flip(0)[0]), rng(ei.flip(0)[1])
    print(f"[诊断] 原始列范围 lnc_col={a0} mi_col={a1} | 交换后 lnc_col={b0} mi_col={b1}")
    raise ValueError("无法规范化交互边。可能混入非法索引或分隔问题。请检查 train2.txt。")

def fix_same_edges(ei, N):
    if ei.numel() == 0: return ei
    if ei.min().item() >= 0 and ei.max().item() < N: return ei
    if ei.min().item() >= 1 and ei.max().item() <= N: return ei - 1
    raise ValueError("同类边索引越界，且非 1-based；请检查 KNN 边与特征行序。")

def check_range(name, ei, src_n, dst_n):
    if ei.numel() == 0: return
    smax = int(ei[0].max().item()); dmax = int(ei[1].max().item())
    assert smax < src_n, f"{name} 源越界: max={smax}, 允许 0..{src_n-1}"
    assert dmax < dst_n, f"{name} 目标越界: max={dmax}, 允许 0..{dst_n-1}"

# ===== 读取特征 =====
FM_lnc = pd.read_csv(LNC_FEAT, index_col=0).values
FM_mi  = pd.read_csv(MI_FEAT,  index_col=0).values
Ln, Mn = FM_lnc.shape[0], FM_mi.shape[0]
print(f"lnc节点数={Ln}, mi节点数={Mn}")

# ===== 读取边 =====
train_raw   = load_array_anysep(TRAIN_TXT)
lnc_knn_raw = load_edges_csv(LNC_KNN)
mi_knn_raw  = load_edges_csv(MI_KNN)

# 辅助打印原始范围
def rng(t): return int(t.min().item()), int(t.max().item())
print(f"train.txt 原始两列范围: col0={rng(train_raw[0])}, col1={rng(train_raw[1])}")

# ===== 规范化 =====
train_edges = normalize_inter_edges(train_raw, Ln, Mn)
lnc_knn = symm(fix_same_edges(lnc_knn_raw, Ln))
mi_knn  = symm(fix_same_edges(mi_knn_raw,  Mn))

# ===== 构图 =====
data = HeteroData()
data['lnc'].x = torch.tensor(FM_lnc, dtype=torch.float)
data['mi'].x  = torch.tensor(FM_mi,  dtype=torch.float)

data['lnc','interacts','mi'].edge_index = train_edges
data['mi','rev_interacts','lnc'].edge_index = train_edges.flip(0)
data['lnc','similar','lnc'].edge_index = lnc_knn
data['mi','similar','mi'].edge_index   = mi_knn

# ===== 自检 =====
check_range("('lnc','interacts','mi')", data['lnc','interacts','mi'].edge_index, Ln, Mn)
check_range("('mi','rev_interacts','lnc')", data['mi','rev_interacts','lnc'].edge_index, Mn, Ln)
check_range("('lnc','similar','lnc')",     data['lnc','similar','lnc'].edge_index,     Ln, Ln)
check_range("('mi','similar','mi')",       data['mi','similar','mi'].edge_index,       Mn, Mn)

def stat(ei): return 0 if ei is None else ei.size(1)
print(f"interacts: {stat(data['lnc','interacts','mi'].edge_index)}")
print(f"rev_interacts: {stat(data['mi','rev_interacts','lnc'].edge_index)}")
print(f"lnc-similar(sym): {stat(data['lnc','similar','lnc'].edge_index)}")
print(f"mi-similar(sym): {stat(data['mi','similar','mi'].edge_index)}")

# ===== 保存 =====
os.makedirs(os.path.dirname(OUT_PT), exist_ok=True)
torch.save(data, OUT_PT)
print(f"saved -> {OUT_PT}")
