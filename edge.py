# build_knn_and_train.py
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph

# ===== 配置 =====
BASE = r"D:\pytorchProject\RR\dateset"   # 数据目录
K = 4                                      # KNN 的 k

# 输入
LNC_FEAT = os.path.join(BASE, "lncrna_final_features.csv")
MI_FEAT  = os.path.join(BASE, "mirna_final_features.csv")
INTER_CSV = os.path.join(BASE, "mirna_lncrna_interaction.csv")

# 输出
LNC_EDGES_CSV = os.path.join(BASE, "lnc_knn_edges.csv")
MI_EDGES_CSV  = os.path.join(BASE, "mi_knn_edges.csv")
TRAIN_TXT     = os.path.join(BASE, "train.txt")
LNC_MAP_CSV   = os.path.join(BASE, "lnc_index_map.csv")
MI_MAP_CSV    = os.path.join(BASE, "mirna_index_map.csv")

def build_knn_edges(feat_csv, k):
    X = pd.read_csv(feat_csv, index_col=0).values
    G = kneighbors_graph(X, k, mode='connectivity', include_self=False)
    coo = G.tocoo()
    edges = np.vstack([coo.row, coo.col]).T.astype(np.int64)
    return edges

def main():
    # 1) 生成 KNN 边
    edges_lnc = build_knn_edges(LNC_FEAT, K)
    pd.DataFrame(edges_lnc).to_csv(LNC_EDGES_CSV, index=False, header=False)
    print(f"lnc edges: {edges_lnc.shape} -> {LNC_EDGES_CSV}")

    edges_mi = build_knn_edges(MI_FEAT, K)
    pd.DataFrame(edges_mi).to_csv(MI_EDGES_CSV, index=False, header=False)
    print(f"mi edges: {edges_mi.shape} -> {MI_EDGES_CSV}")

    # 2) 由交互表生成 train.txt 与索引映射
    df = pd.read_csv(INTER_CSV)
    lnc_set = df['lncrna'].unique().tolist()
    mi_set  = df['mirna'].unique().tolist()

    lnc_map = {name: idx for idx, name in enumerate(lnc_set)}
    mi_map  = {name: idx for idx, name in enumerate(mi_set)}

    df['lnc_idx'] = df['lncrna'].map(lnc_map)
    df['mi_idx']  = df['mirna'].map(mi_map)

    df[['lnc_idx', 'mi_idx']].to_csv(TRAIN_TXT, index=False, header=False)
    pd.DataFrame.from_dict(lnc_map, orient='index').to_csv(LNC_MAP_CSV, header=['lnc_idx'])
    pd.DataFrame.from_dict(mi_map, orient='index').to_csv(MI_MAP_CSV, header=['mi_idx'])

    print(f"train pairs: {len(df)} -> {TRAIN_TXT}")
    print(f"maps -> {LNC_MAP_CSV}, {MI_MAP_CSV}")

if __name__ == "__main__":
    main()
