import os
import subprocess
import csv
import logging
import re
from multiprocessing import Pool, cpu_count

# 配置日志
logging.basicConfig(
    filename='rna_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MAX_LEN = 1000  # 最大子序列长度

def read_rna_sequences(file_path):
    """
    读取 CSV，每行 name,sequence；
    如果同名 RNA 出现多次，只保留第一次；
    且把 T->U，并转换成大写。
    """
    rna_sequences = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            name, sequence = line.split(',')
            sequence = sequence.strip().upper().replace('T', 'U')
            if name not in rna_sequences:
                rna_sequences[name] = sequence
    logging.info(f"Loaded {len(rna_sequences)} unique sequences from {file_path}")
    return rna_sequences

def predict_secondary_structure_for_subsequence(rna_seq, timeout=30):
    """
    调用 RNAfold --noPS，返回该子序列的 bracket + MFE
    """
    cmd = ['RNAfold', '--noPS']
    try:
        proc = subprocess.Popen(cmd,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, _ = proc.communicate(input=rna_seq.encode(), timeout=timeout)
        lines = out.decode().strip().split('\n')
        return lines[1].strip() if len(lines) > 1 else ''
    except subprocess.TimeoutExpired:
        logging.error(f"Timeout on subsequence: {rna_seq[:10]}...")
        return ''

def predict_secondary_structure(rna_seq, timeout=30, max_length=MAX_LEN):
    """
    如果序列太长就分段预测，再拼接结果。
    """
    if len(rna_seq) <= max_length:
        return predict_secondary_structure_for_subsequence(rna_seq, timeout)
    logging.info(f"Sequence too long ({len(rna_seq)}), splitting...")
    parts = [rna_seq[i:i + max_length] for i in range(0, len(rna_seq), max_length)]
    results = [predict_secondary_structure_for_subsequence(s, timeout) for s in parts]
    return ''.join(filter(None, results))

def process_rna_sequence(name, seq, timeout=30):
    """
    调用 predict，解析结构和 MFE。
    """
    struct_line = predict_secondary_structure(seq, timeout)
    if not struct_line:
        return None
    parts = struct_line.split()
    struct_str = parts[0]
    mfe_match = re.search(r'[-+]?\d*\.\d+', struct_line)
    mfe = float(mfe_match.group()) if mfe_match else 0.0
    return {'RNA_Name': name, 'Structure': struct_str, 'Energy': mfe}

def save_rna_structures(output_file, rna_data, timeout=30):
    """
    并行处理所有序列，写入 CSV。
    动态使用 CPU_COUNT-1 的进程数。
    """
    n_workers = max(1, cpu_count() - 1)
    fieldnames = ['RNA_Name', 'Structure', 'Energy']
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        with Pool(processes=n_workers) as pool:
            args = [(name, seq, timeout) for name, seq in rna_data.items()]
            for result in pool.starmap(process_rna_sequence, args, chunksize=10):
                if result:
                    writer.writerow(result)
    print(f"结构摘要已保存到: {output_file}")

if __name__ == '__main__':
    # 自动定位到项目 dateset 文件夹
    HERE = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.abspath(os.path.join(HERE, '..', 'dateset'))

    lncRNA_file = os.path.join(DATASET_DIR, 'lncrna_sequences.csv')
    miRNA_file  = os.path.join(DATASET_DIR, 'mirna_sequences.csv')
    out_lnc = os.path.join(DATASET_DIR, 'lncrna_structures.csv')
    out_mi  = os.path.join(DATASET_DIR, 'mirna_structures.csv')

    lnc_dict = read_rna_sequences(lncRNA_file)
    mi_dict  = read_rna_sequences(miRNA_file)

    save_rna_structures(out_lnc, lnc_dict)
    save_rna_structures(out_mi,  mi_dict)
