import pandas as pd
from sklearn.decomposition import PCA

# === 步骤 1: 加载序列特征 ===
lnc_feat = pd.read_csv("D:/pytorchProject/RR/dateset/lncrna_features.csv", index_col=0)
mir_feat = pd.read_csv("D:/pytorchProject/RR/dateset/mirna_features.csv", index_col=0)

# === 去重：按 RNA 名称（index）去除重复行 ===
lnc_feat = lnc_feat[~lnc_feat.index.duplicated(keep='first')]
mir_feat = mir_feat[~mir_feat.index.duplicated(keep='first')]

# === 步骤 2: 加载结构特征 ===
lnc_struct = pd.read_csv("D:/pytorchProject/RR/dateset/lncrna_structures.csv", index_col=0)
mir_struct = pd.read_csv("D:/pytorchProject/RR/dateset/mirna_structures.csv", index_col=0)

# === 同样去重 ===
lnc_struct = lnc_struct[~lnc_struct.index.duplicated(keep='first')]
mir_struct = mir_struct[~mir_struct.index.duplicated(keep='first')]

def extract_structure_stats(df):
    def calc_features(row):
        struct = row.get('Structure', '') or ''
        total = len(struct)
        count_dot = struct.count('.')
        count_left = struct.count('(')
        count_right = struct.count(')')
        mfe = float(row['Energy']) if 'Energy' in row and pd.notna(row['Energy']) else 0.0
        return pd.Series([
            count_dot / total if total > 0 else 0,
            count_left / total if total > 0 else 0,
            count_right / total if total > 0 else 0,
            abs(count_left - count_right) / total if total > 0 else 0,
            mfe
        ])

    stats_df = df.apply(calc_features, axis=1)
    stats_df.columns = ['dot_ratio', 'left_ratio', 'right_ratio', 'pairing_imbalance', 'mfe']
    stats_df.index = df.index
    return stats_df

# === 步骤 4: 合并特征 ===
def merge_features(seq_feat, struct_feat):
    # 左连接：保留所有序列特征，缺失结构特征时填默认
    merged = seq_feat.join(extract_structure_stats(struct_feat), how='left')
    # 填充缺失的数值列
    if 'mfe' in merged.columns:
        merged['mfe'].fillna(0, inplace=True)
    # 填充缺失的其它结构统计列
    for col in ['dot_ratio', 'left_ratio', 'right_ratio', 'pairing_imbalance']:
        if col in merged.columns:
            merged[col].fillna(0, inplace=True)
    return merged

# === 步骤 5: 构建最终特征矩阵 ===
final_lnc = merge_features(lnc_feat, lnc_struct)
final_mir = merge_features(mir_feat, mir_struct)

# === 再次去重保险（index） ===
final_lnc = final_lnc[~final_lnc.index.duplicated(keep='first')]
final_mir = final_mir[~final_mir.index.duplicated(keep='first')]

# === 步骤 6: 保存 ===
final_lnc.to_csv("D:/pytorchProject/RR/dateset/lncrna_final_features.csv")
final_mir.to_csv("D:/pytorchProject/RR/dateset/mirna_final_features.csv")

print("融合特征（已去重且填充缺失结构值）已保存。")