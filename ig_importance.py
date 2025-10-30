import torch
import pandas as pd
import numpy as np
from torch import nn
from captum.attr import IntegratedGradients
# 假设你的 HetGNN 模型定义在 'test_sj_mk_5_2mwa.py' 文件中
from model import HetGNN

# ======== 1. 环境与参数设置 ========
MODEL_PATH = "D:\pytorchProject\RR\dateset\model_run_50.pth"
GRAPH_PATH = "D:\pytorchProject\RR\dateset\cfhan_hetero_graph.pt"
OUTPUT_DIR = "D:\pytorchProject\RR\dateset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOPK = None  # 如果需要筛选 top-k 特征，请设置一个整数，例如 20

print(f"正在使用的设备: {DEVICE}")
# =================================

# ======== 2. 加载数据 & 模型 ========
# 加载图数据
data = torch.load(GRAPH_PATH)
# 关键修复1：将整个图数据对象移动到目标设备
data = data.to(DEVICE)
print("图数据已成功加载并移动到设备。")

# 准备模型输入维度
in_ch = {
    'lnc': data['lnc'].x.size(1),
    'mi': data['mi'].x.size(1)
}

# 初始化模型并移动到设备
model = HetGNN(in_ch, hidden_dim=128, heads=8).to(DEVICE)
# 加载预训练权重
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("模型已成功加载。")

# ======== 3. 准备 Integrated Gradients 的输入 ========
lnc_x = data['lnc'].x.clone().detach().requires_grad_(True)
mi_x  = data['mi'].x.clone().detach().requires_grad_(True)

# 创建所有可能的 (lnc, mi) 对用于模型预测
N_lnc = lnc_x.shape[0]
N_mi  = mi_x.shape[0]
dummy_pairs = torch.tensor([[i, j] for i in range(N_lnc) for j in range(N_mi)], dtype=torch.long, device=DEVICE)

# ======== 4. 定义用于 IG 的前向传播函数 ========
# 这个函数包装了模型调用，使其符合 Captum 的要求
def model_forward(lnc_input, mi_input):
    out = model({'lnc': lnc_input, 'mi': mi_input}, data.edge_index_dict, dummy_pairs)
    # 关键修复2：使用 .unsqueeze(0) 确保输出是可索引的 1D 张量
    return out.mean().unsqueeze(0)

# ======== 5. 计算特征重要性 ========
print("开始计算 Integrated Gradients...")
ig = IntegratedGradients(model_forward)
attributions, delta = ig.attribute(inputs=(lnc_x, mi_x), return_convergence_delta=True)
print("计算完成。")

# ！！！关键修复3：delta 可能是一个多元素张量，我们应该报告它的汇总统计信息，例如平均值。
print(f"Convergence Delta (mean): {delta.mean().item()}")

# ======== 6. 处理并保存结果 ========
# 分别提取 lnc 和 mi 的归因分数
attributions_lnc = attributions[0]
attributions_mi = attributions[1]

# 计算每个特征维度的平均重要性
imp_lnc = attributions_lnc.abs().mean(dim=0).detach().cpu().numpy()
imp_mi  = attributions_mi.abs().mean(dim=0).detach().cpu().numpy()

# 创建 DataFrame 以便保存
lnc_df = pd.DataFrame({
    "Feature_Index": list(range(len(imp_lnc))),
    "Importance": imp_lnc
})
mi_df = pd.DataFrame({
    "Feature_Index": list(range(len(imp_mi))),
    "Importance": imp_mi
})

# 按重要性排序并筛选 top-k
if isinstance(TOPK, int):
    lnc_df = lnc_df.sort_values("Importance", ascending=False).head(TOPK)
    mi_df = mi_df.sort_values("Importance", ascending=False).head(TOPK)
else:
    lnc_df = lnc_df.sort_values("Importance", ascending=False)
    mi_df = mi_df.sort_values("Importance", ascending=False)


# 保存到 CSV 文件
lnc_path = f"{OUTPUT_DIR}/ig_importance_lnc.csv"
mi_path = f"{OUTPUT_DIR}/ig_importance_mi.csv"
lnc_df.to_csv(lnc_path, index=False)
mi_df.to_csv(mi_path, index=False)

print(f"lncRNA 特征重要性已保存至: {lnc_path}")
print(f"miRNA 特征重要性已保存至: {mi_path}")