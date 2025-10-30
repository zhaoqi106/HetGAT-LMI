# analyze_importance.py
import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

# 设置全局字体和样式
rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 18

# ======== 可根据你的特征拼接顺序自定义 ========
NODE_DIM = 163
SEGMENTS = {
    'kmer': (0, 64),
    'gap_kmer': (64, 128),
    'CTD': (128, 158),
    'struct_MFE': (158, 163),
}


# ============================================

def parse_args():
    ap = argparse.ArgumentParser(description="聚合特征重要性到家族/侧别并出图")
    ap.add_argument("--input", type=str, default=None, help="单个合并文件（含 lnc 与 mi 行，列：Feature, Importance）")
    ap.add_argument("--lnc", type=str, default='D:\pytorchProject\RR\dateset\ig_importance_lnc.csv',
                    help="仅 lnc 的重要性文件")
    ap.add_argument("--mi", type=str, default='D:\pytorchProject\RR\dateset\ig_importance_mi.csv',
                    help="仅 mi  的重要性文件")
    ap.add_argument("--outdir", type=str, default="../pic", help="输出目录")
    ap.add_argument("--node-dim", type=int, default=NODE_DIM, help="每侧特征维度（默认 163）")
    ap.add_argument("--no-plots", action="store_true", help="不生成图片（仅导出 CSV）")
    return ap.parse_args()


def find_default_inputs(args):
    if not args.input and not (args.lnc and args.mi):
        candidates = [
            "shap_feature_importance_fixed.csv",
            "permutation_feature_importance_fixed.csv",
            "feature_importance.csv"
        ]
        for c in candidates:
            if os.path.exists(c):
                args.input = c
                print(f"ℹ️ 发现单文件：{c}")
                return args
        if os.path.exists("ig_importance_lnc.csv") and os.path.exists("ig_importance_mi.csv"):
            args.lnc, args.mi = "ig_importance_lnc.csv", "ig_importance_mi.csv"
            print("ℹ️ 发现双文件：ig_importance_lnc.csv + ig_importance_mi.csv")
    return args


def load_single_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    score_cols = [c for c in df.columns if "importance" in c.lower()]
    if not score_cols:
        raise ValueError(f"{path} 找不到 Importance 列")
    score_col = score_cols[0]
    keep = df[["Feature", score_col]].copy()
    keep.rename(columns={score_col: "Importance"}, inplace=True)
    return keep


def load_inputs(args):
    if args.input:
        return load_single_csv(args.input)
    else:
        if not (args.lnc and args.mi):
            raise ValueError("未提供 --input，也未同时提供 --lnc 与 --mi")
        lnc_df = load_single_csv(args.lnc)
        mi_df = load_single_csv(args.mi)

        def ensure_prefix(df, side_prefix):
            def _ensure(name):
                s = str(name)
                low = s.lower()
                if low.startswith("lnc") or low.startswith("mi"):
                    return s
                return f"{side_prefix}_feat_{s}" if re.search(r"\d+$", s) else f"{side_prefix}_{s}"

            df = df.copy()
            df["Feature"] = df["Feature"].map(_ensure)
            return df

        lnc_df = ensure_prefix(lnc_df, "lnc")
        mi_df = ensure_prefix(mi_df, "mi")
        return pd.concat([lnc_df, mi_df], axis=0, ignore_index=True)


def idx2family(idx, node_dim, segments):
    for fam, (lo, hi) in segments.items():
        if lo <= idx < hi:
            return fam
    return "unknown"


def parse_feature_name(name):
    s = str(name).strip()
    low = s.lower()
    if low.startswith("mi"):
        side = "mi"
    elif low.startswith("lnc"):
        side = "lnc"
    else:
        if "mirna" in low:
            side = "mi"
        elif "lncrna" in low:
            side = "lnc"
        else:
            side = "unknown"
    m = re.search(r'(\d+)$', s)
    idx = int(m.group(1)) if m else None
    return side, idx


def aggregate(df, node_dim, segments):
    rows = []
    for _, r in df.iterrows():
        feat = r["Feature"]
        imp = float(r["Importance"])
        side, idx = parse_feature_name(feat)
        if side not in {"lnc", "mi"} or idx is None:
            continue
        fam = idx2family(idx % node_dim, node_dim, segments)
        if fam == "unknown":
            continue
        rows.append({"side": side, "family": fam, "feature": feat, "score": imp})
    clean = pd.DataFrame(rows)
    return clean


def save_family_contrib(clean, outdir):
    g = clean.groupby(["side", "family"])["score"].sum().reset_index()
    total = g["score"].sum() + 1e-12
    g["percent_%"] = 100.0 * g["score"] / total
    g = g.sort_values(["side", "percent_%"], ascending=[True, False])
    out_path = os.path.join(outdir, "family_contributions.csv")
    g.to_csv(out_path, index=False)
    print(f"✅ 保存家族占比: {out_path}")
    return g


def save_top_features_by_family(clean, outdir, k=10):
    rows = []
    for side in ["lnc", "mi"]:
        for fam in SEGMENTS.keys():
            sub = clean[(clean["side"] == side) & (clean["family"] == fam)]
            if sub.empty:
                continue
            sub = sub.sort_values("score", ascending=False).head(k)
            for _, r in sub.iterrows():
                rows.append({
                    "side": side,
                    "family": fam,
                    "feature": r["feature"],
                    "score": r["score"]
                })
    topdf = pd.DataFrame(rows)
    out_path = os.path.join(outdir, "top_features_by_family.csv")
    topdf.to_csv(out_path, index=False)
    print(f"✅ 保存各家族 Top 特征: {out_path}")
    return topdf


def plot_family_bars(g, outdir):
    """生成家族贡献图，使用与Top20图一致的风格和色系"""
    sns.set_style("white")

    for side in ["lnc", "mi"]:
        sub = g[g["side"] == side].sort_values("percent_%", ascending=True)
        if sub.empty:
            continue

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 使用不同的蓝绿色系 - 确保浅蓝色在顶部
        # 对于LNC使用蓝绿色系
        if side == "lnc":
            colors = ['#377790', '#639fba', '#a2d3e5', '#c4dce8']  # 浅蓝到深蓝
        # 对于MI使用蓝紫色系
        else:
            colors = ['#377790', '#639fba', '#a2d3e5', '#c4dce8']  # 浅蓝到紫

        # 确保颜色数量与数据点匹配
        if len(sub) <= len(colors):
            bar_colors = colors[:len(sub)]
        else:
            # 如果需要更多颜色，使用渐变色
            cmap = LinearSegmentedColormap.from_list('custom_blue', ['#126782', '#E6F5FF'])
            bar_colors = [cmap(i / (len(sub) - 1)) for i in range(len(sub))]

        # 绘制水平条形图 -
        bars = ax.barh(sub["family"], sub["percent_%"],
                       color=bar_colors, edgecolor='none', linewidth=0.8, alpha=0.85)

        # 添加数值标签 - 确保显示足够的精度
        for i, (bar, value) in enumerate(zip(bars, sub["percent_%"])):
            # 如果值非常小，使用科学计数法显示
            if value < 0.01:
                label_text = f'{value:.2e}%'
            else:
                label_text = f'{value:.2f}%'

            ax.text(bar.get_width() + max(sub["percent_%"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    label_text,
                    ha='left', va='center', fontsize=11,
                    fontweight='bold', color='#0077B6')

        # 设置标签和标题
        ax.set_xlabel("Contribution Percentage (%)", fontsize=14, fontweight='bold', color='#333333')
        ax.set_ylabel("Feature Family", fontsize=14, fontweight='bold', color='#333333')
        ax.set_title(f"{side.upper()} RNA - Family Contribution",
                     fontsize=16, fontweight='bold', pad=20, color='#333333')

        # 美化图形
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # 设置轴线颜色
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1.0)

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.4, axis='x', color='#dddddd')

        # 设置刻度颜色
        ax.tick_params(axis='x', colors='#666666')
        ax.tick_params(axis='y', colors='#666666')

        # 添加背景色
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')

        # 调整布局
        plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

        # 保存高质量图像
        path = os.path.join(outdir, f"family_{side}.png")
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"✅ 保存家族占比图: {path}")


def plot_shap_style_top20(clean, outdir):
    """生成SHAP风格的Top20特征图"""
    sns.set_style("white")

    for side in ["lnc", "mi"]:
        sub = clean[clean["side"] == side].sort_values("score", ascending=False).head(20)
        if sub.empty:
            continue

        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 10))

        # 使用蓝绿渐变色系 - 与家族贡献图不同
        blue_green = LinearSegmentedColormap.from_list('blue_green', ['#009F6B', '#0077B6', '#E6F5FF'])
        colors = blue_green(np.linspace(0.2, 0.8, len(sub)))

        # 绘制水平条形图 - SHAP风格
        y_pos = np.arange(len(sub))
        bars = ax.barh(y_pos, sub["score"], color=colors, edgecolor='none', linewidth=0.8, alpha=0.85)

        # 创建家族标签映射
        family_labels = {
            'kmer': 'mi_kmer',
            'gap_kmer': 'mi_gap',
            'CTD': 'mi_ctd',
            'struct_MFE': 'mi_struct'
        }

        # 设置y轴标签 - 使用家族名称而不是特征名称
        y_labels = [family_labels.get(fam, fam) for fam in sub["family"]]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=12, fontweight='bold', color='#333333')

        # 设置标签和标题
        ax.set_xlabel("Feature Importance Score (×10⁻⁵)", fontsize=16, fontweight='bold', color='#333333')
        ax.set_ylabel("Feature Family", fontsize=16, fontweight='bold', color='#333333')
        ax.set_title(f"MI RNA - Top 20 Most Important Features",
                     fontsize=20, fontweight='bold', pad=20, color='#333333')

        # 设置x轴格式，使用科学计数法显示小数值
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-5, 5))
        ax.xaxis.set_major_formatter(formatter)

        # 添加数值标签到条形图右侧
        for i, (bar, value) in enumerate(zip(bars, sub["score"])):
            # 将数值标签放在条形图右侧
            ax.text(bar.get_width() + max(sub["score"]) * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:.2e}',
                    ha='left', va='center', fontsize=10,
                    fontweight='bold', color='#0077B6')

        # 美化图形
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

        # 设置轴线颜色
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(1.0)

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.4, axis='x', color='#dddddd')

        # 设置刻度颜色
        ax.tick_params(axis='x', colors='#666666')
        ax.tick_params(axis='y', colors='#666666')

        # 添加背景色
        ax.set_facecolor('#fafafa')
        fig.patch.set_facecolor('white')

        # 调整布局
        plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)

        # 保存高质量图像
        path = os.path.join(outdir, f"shap_summary_{side}.png")
        plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"✅ 保存SHAP摘要图: {path}")


def plot_gate_weighted_family(fam, outdir):
    """生成Gate加权的家族贡献图"""
    sns.set_style("white")

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 使用蓝紫色系 - 与家族贡献图和Top20图不同
    blue_purple = ['#E6F5FF', '#8ECAE6', '#5E60CE', '#7400B8', '#560BAD']

    # 确保颜色数量与数据点匹配
    if len(fam) <= len(blue_purple):
        colors = blue_purple[:len(fam)]
    else:
        # 如果需要更多颜色，使用渐变色
        cmap = LinearSegmentedColormap.from_list('blue_purple', ['#E6F5FF', '#560BAD'])
        colors = [cmap(i / (len(fam) - 1)) for i in range(len(fam))]

    # 绘制水平条形图
    bars = ax.barh(fam["family"], fam["global_percent"],
                   color=colors, edgecolor='none', linewidth=0.8, alpha=0.85)

    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, fam["global_percent"])):
        # 如果值非常小，使用科学计数法显示
        if value < 0.01:
            label_text = f'{value:.2e}%'
        else:
            label_text = f'{value:.2f}%'

        ax.text(bar.get_width() + max(fam["global_percent"]) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                ha='left', va='center', fontsize=11,
                fontweight='bold', color='#5E60CE')

    # 设置标签和标题
    ax.set_xlabel("Global Contribution Percentage (%)", fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel("Feature Family", fontsize=14, fontweight='bold', color='#333333')
    ax.set_title("Gate-weighted Family Contributions",
                 fontsize=16, fontweight='bold', pad=20, color='#333333')

    # 美化图形
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # 设置轴线颜色
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.4, axis='x', color='#dddddd')

    # 设置刻度颜色
    ax.tick_params(axis='x', colors='#666666')
    ax.tick_params(axis='y', colors='#666666')

    # 添加背景色
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')

    # 调整布局
    plt.subplots_adjust(left=0.2, right=0.95, top=0.9, bottom=0.15)

    # 保存高质量图像
    path = os.path.join(outdir, "family_percent_gate_weighted.png")
    plt.savefig(path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"✅ 保存Gate加权家族占比图: {path}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    args = find_default_inputs(args)

    # 1) 读取
    df = load_inputs(args)
    # 2) 规整并聚合
    clean = aggregate(df, node_dim=args.node_dim, segments=SEGMENTS)
    if clean.empty:
        raise RuntimeError("没有解析出有效的 (side, family, feature, score) 行，请检查输入的 Feature 命名。")

    # 3) 家族占比 + 导出
    g = save_family_contrib(clean, args.outdir)
    # 4) 导出各家族 Top-特征
    _ = save_top_features_by_family(clean, args.outdir, k=10)

    # 5) 画图
    if not args.no_plots:
        plot_family_bars(g, args.outdir)
        plot_shap_style_top20(clean, args.outdir)

        # 6) 生成Gate加权的家族贡献图
        # 这里需要计算Gate加权的全局贡献
        # 假设我们已经有了fam DataFrame，包含family和global_percent列
        # 如果没有，可以从家族贡献数据中计算
        # 这里只是示例，您需要根据实际情况调整
        try:
            # 尝试加载gate加权的数据
            gate_weighted_path = os.path.join(args.outdir, "family_percent_gate_weighted.csv")
            if os.path.exists(gate_weighted_path):
                fam = pd.read_csv(gate_weighted_path)
                plot_gate_weighted_family(fam, args.outdir)
            else:
                print("⚠ 未找到Gate加权数据，跳过生成Gate加权家族贡献图")
        except Exception as e:
            print(f"⚠ 生成Gate加权家族贡献图时出错: {e}")

    # 控制台摘要
    print("\n== 家族贡献（百分比）==")
    print(g[["side", "family", "percent_%"]].to_string(index=False))

    print("\n🎉 完成！输出文件：")
    print(f"  - {os.path.join(args.outdir, 'family_contributions.csv')}")
    print(f"  - {os.path.join(args.outdir, 'top_features_by_family.csv')}")
    if not args.no_plots:
        print(f"  - {os.path.join(args.outdir, 'family_lnc.png')} / {os.path.join(args.outdir, 'family_mi.png')}")
        print(
            f"  - {os.path.join(args.outdir, 'shap_summary_lnc.png')} / {os.path.join(args.outdir, 'shap_summary_mi.png')}")
        print(f"  - {os.path.join(args.outdir, 'family_percent_gate_weighted.png')}")


if __name__ == "__main__":
    main()