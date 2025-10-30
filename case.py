# case_batch_loeo.py  —— 批量LOEO（284个lncRNA）
import os, json, random
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

# 固定随机
random.seed(0); np.random.seed(0); torch.manual_seed(0)

# ===== 路径与超参 =====
BASE = r"..\dateset"   # 如在Win上改成 D:\pytorchProject\RR\dateset
GRAPH_PT = os.path.join(BASE, "cfhan_hetero_graph.pt")
INTER_CSV = os.path.join(BASE, "mirna_lncrna_interaction.csv")
OUT_DIR = os.path.join(BASE, "case_outputs_loeo_all")
os.makedirs(OUT_DIR, exist_ok=True)

SIDE = "lnc"         # 本次对 lnc 做LOEO
TOPK = 20
EPOCHS = 200
LR = 1e-4
PATIENCE = 30
BATCH_POS = 4096
NEG_RATIO = 1
HEADS = 8
HID = 128

# ===== 模型 =====
from model import HetGNN  # 保持与你训练一致

# ===== 工具函数 =====
def load_graph_and_maps():
    data = torch.load(GRAPH_PT, map_location="cpu")
    df = pd.read_csv(INTER_CSV)
    lnc_list = df["lncrna"].drop_duplicates().tolist()
    mi_list  = df["mirna"].drop_duplicates().tolist()
    l2i = {n:i for i,n in enumerate(lnc_list)}
    m2i = {n:i for i,n in enumerate(mi_list)}
    pos_set = set((l2i[r.lncrna], m2i[r.mirna]) for _,r in df.iterrows())
    return data, df, lnc_list, mi_list, l2i, m2i, pos_set

def guess_edge_keys(g):
    fwd = [k for k in g.edge_index_dict.keys() if k[0]=="lnc" and k[2]=="mi"]
    bwd = [k for k in g.edge_index_dict.keys() if k[0]=="mi"  and k[2]=="lnc"]
    if not fwd: raise KeyError("未找到 lnc→mi 边键")
    return fwd[0], (bwd[0] if bwd else None)

def build_masked_graph(data, df, l2i, m2i, side, target):
    t_idx = l2i[target] if side=="lnc" else m2i[target]
    if side=="lnc":
        w = df[df["lncrna"]==target][["lncrna","mirna"]].copy()
        w["lnc_idx"] = t_idx; w["mi_idx"] = w["mirna"].map(m2i)
    else:
        w = df[df["mirna"]==target][["lncrna","mirna"]].copy()
        w["mi_idx"] = t_idx; w["lnc_idx"] = w["lncrna"].map(l2i)
    w = w.dropna().astype({"lnc_idx":int,"mi_idx":int})
    if w.empty: return None, None, t_idx

    g = data.clone()
    kf, kb = guess_edge_keys(g)

    def _drop(pair, forward=True):
        dev = pair.device
        keep = torch.ones(pair.size(1), dtype=torch.bool, device=dev)
        if forward:
            wset = set(map(tuple, w[["lnc_idx","mi_idx"]].to_numpy()))
            for e in range(pair.size(1)):
                a,b = int(pair[0,e]), int(pair[1,e])
                if (side=="lnc" and a==t_idx) or (side=="mi" and b==t_idx):
                    if (a,b) in wset: keep[e]=False
        else:
            wset = set(map(tuple, w[["mi_idx","lnc_idx"]].to_numpy()))
            for e in range(pair.size(1)):
                a,b = int(pair[0,e]), int(pair[1,e])
                if (side=="mi" and a==t_idx) or (side=="lnc" and b==t_idx):
                    if (a,b) in wset: keep[e]=False
        return pair[:,keep]

    g.edge_index_dict[kf] = _drop(g.edge_index_dict[kf], True)
    if kb and kb in g.edge_index_dict:
        g.edge_index_dict[kb] = _drop(g.edge_index_dict[kb], False)
    return g, w, t_idx

def sample_pos(df, l2i, m2i, side, target, n):
    sub = df[df["lncrna"]!=target] if side=="lnc" else df[df["mirna"]!=target]
    idx = np.random.randint(0, len(sub), size=n)
    rows = sub.iloc[idx]
    l = torch.tensor(rows["lncrna"].map(l2i).to_numpy(), dtype=torch.long)
    m = torch.tensor(rows["mirna"].map(m2i).to_numpy(), dtype=torch.long)
    y = torch.ones(l.size(0), dtype=torch.float32)
    return l,m,y

def sample_neg(lnc_n, mi_n, pos_set, side, t_idx, n):
    buf=[]
    while len(buf)<n:
        a = np.random.randint(0, lnc_n)
        b = np.random.randint(0, mi_n)
        if (a,b) in pos_set: continue
        if side=="lnc" and a==t_idx: continue
        if side=="mi"  and b==t_idx: continue
        buf.append((a,b))
    arr = np.array(buf, dtype=int)
    l = torch.tensor(arr[:,0], dtype=torch.long)
    m = torch.tensor(arr[:,1], dtype=torch.long)
    y = torch.zeros(l.size(0), dtype=torch.float32)
    return l,m,y

def train_one(g, df, lnc_list, mi_list, l2i, m2i, side, t_idx, target):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = g.to(device)
    in_ch = {'lnc': g['lnc'].x.size(1), 'mi': g['mi'].x.size(1)}
    model = HetGNN(in_ch, hidden_dim=HID, heads=HEADS).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    pos_set = set((l2i[r.lncrna], m2i[r.mirna]) for _,r in df.iterrows())

    best = {"loss": 1e9, "state": None}
    noimp = 0
    for ep in range(1, EPOCHS+1):
        model.train()
        lp, mp, yp = sample_pos(df, l2i, m2i, side, target, BATCH_POS)
        ln, mn, yn = sample_neg(len(lnc_list), len(mi_list), pos_set, side, t_idx, BATCH_POS*NEG_RATIO)
        l = torch.cat([lp, ln]).to(device, non_blocking=True)
        m = torch.cat([mp, mn]).to(device, non_blocking=True)
        y = torch.cat([yp, yn]).to(device, non_blocking=True)
        pairs = torch.stack([l,m], dim=1)
        logits = model(g.x_dict, g.edge_index_dict, pairs)
        loss = bce(logits.view(-1), y)

        opt.zero_grad(); loss.backward(); opt.step()
        cur = float(loss.detach().cpu())
        if cur < best["loss"] - 1e-4:
            best = {"loss": cur, "state": {k:v.cpu() for k,v in model.state_dict().items()}}
            noimp = 0
        else:
            noimp += 1
        if ep % 10 == 0 or ep == 1:
            print(f"[{target}] epoch {ep} loss {cur:.4f} best {best['loss']:.4f}")
        if noimp >= PATIENCE: break

    model.load_state_dict(best["state"])
    return model

@torch.no_grad()
def score_target(model, g, side, t_idx, lnc_list, mi_list):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval(); g = g.to(device)
    if side=="lnc":
        pairs = torch.tensor([[t_idx, j] for j in range(len(mi_list))], dtype=torch.long, device=device)
    else:
        pairs = torch.tensor([[i, t_idx] for i in range(len(lnc_list))], dtype=torch.long, device=device)
    logits = model(g.x_dict, g.edge_index_dict, pairs)
    return torch.sigmoid(logits.view(-1)).cpu().numpy()

def eval_entity(scores, side, t_idx, withheld, lnc_list, mi_list, topk=20):
    if side=="lnc":
        items = [(j, scores[j]) for j in range(len(mi_list))]
        pos_idx = set(withheld["mi_idx"].tolist())
        tgt_name = lnc_list[t_idx]; names = mi_list; axes=("lncRNA","miRNA")
    else:
        items = [(i, scores[i]) for i in range(len(lnc_list))]
        pos_idx = set(withheld["lnc_idx"].tolist())
        tgt_name = mi_list[t_idx]; names = lnc_list; axes=("miRNA","lncRNA")

    items.sort(key=lambda x:x[1], reverse=True)
    ranks = []
    for p in pos_idx:
        r = next(i for i,(idx,_) in enumerate(items) if idx==p) + 1
        ranks.append(r)
    mrr = float(np.mean([1.0/r for r in ranks])) if ranks else float("nan")
    hits = sum(1 for i,(idx,_) in enumerate(items[:topk]) if idx in pos_idx)
    recall_at_k = hits / max(1,len(pos_idx))

    top_rows = []
    for i,(idx,sc) in enumerate(items[:topk], start=1):
        top_rows.append({
            axes[0]: tgt_name, axes[1]: names[idx],
            "score": sc, "rank": i,
            "withheld_positive": 1 if idx in pos_idx else 0
        })
    df_top = pd.DataFrame(top_rows)
    metrics = {
        "target": tgt_name,
        "num_withheld_pos": len(pos_idx),
        "MRR": mrr,
        f"Recall@{topk}": recall_at_k,
        "mean_rank": float(np.mean(ranks)) if ranks else float("nan")
    }
    return df_top, metrics

def run_one(preloaded, target):
    data, df, lnc_list, mi_list, l2i, m2i, _ = preloaded
    g_mask, withheld, t_idx = build_masked_graph(data, df, l2i, m2i, SIDE, target)
    if withheld is None or withheld.empty:
        print(f"[跳过] {target} 无真阳性"); return None

    subdir = os.path.join(OUT_DIR, f"{SIDE}_{target}")
    os.makedirs(subdir, exist_ok=True)
    mfile = os.path.join(subdir, "metrics.json")
    if os.path.exists(mfile):
        print(f"[跳过] 已存在 {mfile}"); return json.load(open(mfile, encoding="utf-8"))

    model = train_one(g_mask, df, lnc_list, mi_list, l2i, m2i, SIDE, t_idx, target)
    scores = score_target(model, g_mask, SIDE, t_idx, lnc_list, mi_list)
    df_top, metrics = eval_entity(scores, SIDE, t_idx, withheld, lnc_list, mi_list, TOPK)

    withheld.to_csv(os.path.join(subdir, "withheld_truth.csv"), index=False)
    df_top.to_csv(os.path.join(subdir, f"top{TOPK}.csv"), index=False)
    with open(mfile, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("done:", target, metrics)
    return metrics

# ===== 主流程 =====
if __name__ == "__main__":
    pre = load_graph_and_maps()
    # 只取“有正样本”的 lncRNA；通常为全部 284 个
    df_all = pd.read_csv(INTER_CSV)
    targets = df_all["lncrna"].value_counts().index.tolist()

    rows = []
    total = len(targets)
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{total}] target={t}")
        m = run_one(pre, t)
        if m: rows.append(m)

    # 汇总
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
        print("summary ->", os.path.join(OUT_DIR, "summary.csv"))
