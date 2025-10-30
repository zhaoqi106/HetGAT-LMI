import pandas as pd
import string
from collections import defaultdict
from itertools import product

# 读取 CSV 文件
df = pd.read_csv("D:\pytorchProject\RR\dateset\mirna_lncrna_interaction.csv")

# === 去重：保证每个 RNA 只出现一次 ===
unique_lnc = df.drop_duplicates(subset="lncrna", keep="first")[["lncrna", "lncrna_seq"]]
unique_mi  = df.drop_duplicates(subset="mirna", keep="first")[["mirna",  "mirna_seq"]]

# 清洗函数
def clean_seq(seq):
    if pd.isna(seq):
        return ""
    seq = seq.translate(str.maketrans('', '', string.punctuation))
    seq = seq.replace('.', '').replace(',', '')
    # 统一大写并把 T 换成 U
    return seq.strip().upper().replace("T", "U")

# 应用清洗
unique_lnc["lncrna_seq"] = unique_lnc["lncrna_seq"].apply(clean_seq)
unique_mi["mirna_seq"]  = unique_mi["mirna_seq"].apply(clean_seq)

# === 再次去重：防止同名 RNA 多次出现（按名称去重） ===
unique_lnc = unique_lnc.drop_duplicates(subset=["lncrna"], keep="first")
unique_mi  = unique_mi.drop_duplicates(subset=["mirna"], keep="first")

print(f"lncRNA：{len(unique_lnc)} 条，miRNA：{len(unique_mi)} 条（已去重）")

# === 新增：导出纯序列文件 ===
unique_lnc.to_csv("D:/pytorchProject/RR/dateset/lncrna_sequences.csv", index=False,
                  columns=["lncrna", "lncrna_seq"])
unique_mi.to_csv("D:/pytorchProject/RR/dateset/mirna_sequences.csv",  index=False,
                  columns=["mirna",  "mirna_seq"])
print("lncrna_sequences.csv 和 mirna_sequences.csv 已保存。")

# ———————— 以下为特征提取部分 ————————

def kmer_features(seq, k=3):
    kmer_counts = defaultdict(int)
    total = len(seq) - k + 1
    for i in range(total):
        kmer = seq[i:i + k]
        kmer_counts[kmer] += 1
    bases = 'AUCG'
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    return [kmer_counts[kmer] / total if total > 0 else 0 for kmer in all_kmers]

def g_gap_features(seq, k=2, g=1):
    gap_kmer_counts = defaultdict(int)
    total = len(seq) - (k - 1) * (g + 1)
    for i in range(total):
        gap_kmer = ''.join([seq[i + j * (g + 1)] for j in range(k)])
        gap_kmer_counts[gap_kmer] += 1
    bases = 'AUCG'
    all_kmers = [''.join(p) for p in product(bases, repeat=k)]
    return [gap_kmer_counts[kmer] / total if total > 0 else 0 for kmer in all_kmers]

def calc_dist(seq, n, base, count):
    indices = [i for i, x in enumerate(seq) if x == base]
    if not indices:
        return [0] * 5
    max_index = len(indices) - 1
    return [indices[min(int(count * x / 4), max_index)] / n for x in range(5)]

def CTD(seq):
    n = len(seq)
    if n == 0:
        return [0] * 30
    num_A, num_U, num_G, num_C = seq.count("A"), seq.count("U"), seq.count("G"), seq.count("C")
    trans_counts = defaultdict(int)
    for i in range(n - 1):
        trans_counts[seq[i:i + 2]] += 1
    A_dist = calc_dist(seq, n, "A", num_A)
    U_dist = calc_dist(seq, n, "U", num_U)
    G_dist = calc_dist(seq, n, "G", num_G)
    C_dist = calc_dist(seq, n, "C", num_C)
    return [
        num_A / n, num_U / n, num_G / n, num_C / n,
        trans_counts["AU"] / (n - 1) if n > 1 else 0,
        trans_counts["AG"] / (n - 1) if n > 1 else 0,
        trans_counts["AC"] / (n - 1) if n > 1 else 0,
        trans_counts["UG"] / (n - 1) if n > 1 else 0,
        trans_counts["UC"] / (n - 1) if n > 1 else 0,
        trans_counts["GC"] / (n - 1) if n > 1 else 0,
    ] + A_dist + U_dist + G_dist + C_dist

def extract_features(data, seq_column, name_column):
    all_features = []
    # 首列为 RNA 名称，再后面是各类特征
    columns = [name_column] + [f'kmer_{i}' for i in range(64)] + \
              [f'g_gap_{i}' for i in range(64)] + [f'CTD_{i}' for i in range(30)]

    for _, row in data.iterrows():
        seq = row[seq_column]
        feats = kmer_features(seq, 3) + g_gap_features(seq, 3, 1) + CTD(seq)
        all_features.append([row[name_column]] + feats)

    feature_df = pd.DataFrame(all_features, columns=columns)
    # === 去重：按 RNA 名称保证唯一 ===
    feature_df = feature_df.drop_duplicates(subset=[name_column], keep="first")
    return feature_df

# 执行并保存特征
lncRNA_features = extract_features(unique_lnc, "lncrna_seq", "lncrna")
miRNA_features = extract_features(unique_mi,  "mirna_seq",  "mirna")

lncRNA_features.to_csv("D:/pytorchProject/RR/dateset/lncrna_features.csv", index=False)
miRNA_features.to_csv("D:/pytorchProject/RR/dateset/mirna_features.csv",  index=False)

print("Feature extraction 完成，并已按名称去重保存。")
