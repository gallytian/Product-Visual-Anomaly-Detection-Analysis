"""
结果分析脚本：方法对比图表 + 失败案例分析（误检 / 漏检）

用法:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --topn 4

输出（results/analysis/）:
    01_model_comparison.png      各模型指标平均对比柱状图
    02_category_image_auroc.png  各类别 Image AUROC 分组对比
    03_failure_cases_<model>.png 误检 / 漏检案例可视化
    failure_summary.json         误检漏检统计
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT_DIR = RESULTS / "analysis"

METRICS = ["image_auroc", "pixel_auroc", "aupro", "f1_max"]
METRIC_LABELS = {
    "image_auroc": "Image AUROC",
    "pixel_auroc": "Pixel AUROC",
    "aupro": "AUPRO",
    "f1_max": "F1-max",
}
COLORS = {"patchcore": "#667eea", "padim": "#764ba2"}

CMAP = matplotlib.colormaps["jet"]


# ---------- 图表 1: 指标平均对比 ----------
def plot_metric_comparison(summary: pd.DataFrame, out_path: Path):
    models = list(summary["model"].unique())
    mean_df = summary.groupby("model")[METRICS].mean()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=110)
    for ax, metric in zip(axes, METRICS):
        vals = [mean_df.loc[m, metric] * 100 for m in models]
        bars = ax.bar(models, vals, color=[COLORS.get(m, "#888") for m in models], width=0.55)
        ax.set_title(METRIC_LABELS[metric], fontsize=12)
        ax.set_ylim(85, 101)
        ax.set_ylabel("%")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=10)
    fig.suptitle("模型指标对比（5 类别平均）", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path.name)


# ---------- 图表 2: 各类别 Image AUROC ----------
def plot_category_auroc(summary: pd.DataFrame, out_path: Path):
    models = list(summary["model"].unique())
    categories = sorted(summary["category"].unique())
    n = len(categories)
    width = 0.36
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(13, 5), dpi=110)
    for i, m in enumerate(models):
        sub = summary[summary["model"] == m].set_index("category")
        vals = [sub.loc[c, "image_auroc"] * 100 if c in sub.index else 0 for c in categories]
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=m, color=COLORS.get(m, "#888"))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Image AUROC (%)")
    ax.set_ylim(80, 102)
    ax.set_title("各类别 Image AUROC 对比", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out_path.name)


# ---------- 数据加载 ----------
def load_records(models):
    """汇总所有类别的逐样本分数、标签与异常图"""
    records = []
    for model in models:
        model_dir = RESULTS / model
        if not model_dir.is_dir():
            continue
        for cat_dir in sorted(model_dir.iterdir()):
            npz_f = cat_dir / "scores.npz"
            if not cat_dir.is_dir() or not npz_f.exists():
                continue
            thr = None
            mfile = cat_dir / "metrics.json"
            if mfile.exists():
                thr = json.loads(mfile.read_text(encoding="utf-8")).get("f1_threshold")
            data = np.load(npz_f, allow_pickle=True)
            for s, l, m, p in zip(data["scores"], data["labels"], data["maps"], data["paths"]):
                records.append({
                    "model": model,
                    "category": cat_dir.name,
                    "score": float(s),
                    "label": int(l),
                    "map": np.asarray(m, dtype=np.float32),
                    "path": str(p),
                    "threshold": float(thr) if thr is not None else None,
                })
    return records


def gt_mask_path(img_path: Path) -> Path:
    """由测试图路径推导 GT 掩码路径"""
    parts = list(img_path.parts)
    try:
        i = parts.index("test")
    except ValueError:
        return None
    defect = parts[i + 1]
    name = parts[-1]
    stem = Path(name).stem
    suffix = Path(name).suffix
    mask_name = f"{stem}_mask{suffix}"
    return Path(*parts[:i]) / "ground_truth" / defect / mask_name


def _load_display(img_path: Path, hw):
    """加载原图并缩放到与异常图相同尺寸"""
    img = Image.open(img_path).convert("RGB")
    w, h = hw[1], hw[0]
    img = img.resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _norm(amap):
    lo, hi = float(amap.min()), float(amap.max())
    return np.zeros_like(amap) if hi - lo < 1e-8 else (amap - lo) / (hi - lo)


def plot_failure_cases(records, model, topn, out_path):
    """生成误检 / 漏检案例对比图"""
    recs = [r for r in records if r["model"] == model and r["threshold"] is not None]
    fp = sorted([r for r in recs if r["label"] == 0 and r["score"] > r["threshold"]],
                key=lambda r: -r["score"])[:topn]
    fn = sorted([r for r in recs if r["label"] == 1 and r["score"] <= r["threshold"]],
                key=lambda r: r["score"])[:topn]
    if not fp and not fn:
        print(f"  {model}: 无失败案例")
        return {"false_positive": 0, "false_negative": 0}

    rows = []
    if fp:
        rows.append(("误检（正常图被误判为异常）", fp, False))
    if fn:
        rows.append(("漏检（缺陷图未被检出）", fn, True))

    ncols = topn
    nrows = len(rows) * 2  # 每组两行: 原图 / 热力图
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.3 * nrows), dpi=110)
    axes = np.atleast_2d(axes)
    if axes.shape[0] == 1 and nrows == 1:
        axes = axes.reshape(1, -1)

    for r_i, (title, items, show_gt) in enumerate(rows):
        for c in range(ncols):
            ax_img = axes[r_i * 2][c]
            ax_map = axes[r_i * 2 + 1][c]
            if c >= len(items):
                ax_img.axis("off")
                ax_map.axis("off")
                continue
            r = items[c]
            img_path = Path(r["path"])
            try:
                img = _load_display(img_path, r["map"].shape)
            except Exception:
                ax_img.axis("off")
                ax_map.axis("off")
                continue
            ax_img.imshow(img)
            ax_img.set_title(f"{r['category']} | 分数 {r['score']:.4f}", fontsize=8)
            ax_img.axis("off")
            ax_map.imshow(_norm(r["map"]), cmap="jet")
            ax_map.set_title(f"阈值 {r['threshold']:.4f}", fontsize=8)
            ax_map.axis("off")

            if show_gt:
                mp = gt_mask_path(img_path)
                if mp is not None and mp.exists():
                    mask = np.asarray(Image.open(mp).convert("L").resize(
                        (r["map"].shape[1], r["map"].shape[0]), Image.NEAREST))
                    ax_map.contour(mask > 127, levels=[0.5], colors="w", linewidths=1.0)

        axes[r_i * 2][0].set_ylabel(title, fontsize=10)

    fig.suptitle(f"{model} 失败案例分析（白线为缺陷真值边界）", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path.name}  (误检 {len(fp)}, 漏检 {len(fn)})")
    return {"false_positive": len(fp), "false_negative": len(fn)}


def main():
    parser = argparse.ArgumentParser(description="结果分析与失败案例可视化")
    parser.add_argument("--topn", type=int, default=4, help="每类案例展示数量")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS / "summary.csv"
    if not csv_path.exists():
        print(f"未找到 {csv_path}，请先运行 scripts/run_all.py")
        return
    summary = pd.read_csv(csv_path)

    # 基础对比图表
    plot_metric_comparison(summary, OUT_DIR / "01_model_comparison.png")
    plot_category_auroc(summary, OUT_DIR / "02_category_image_auroc.png")

    # 失败案例分析
    models = list(summary["model"].unique())
    records = load_records(models)
    if not records:
        print("未找到 scores.npz，跳过失败案例分析")
        return

    summary_out = {}
    for m in models:
        stats = plot_failure_cases(records, m, args.topn, OUT_DIR / f"03_failure_cases_{m}.png")
        n_norm = sum(1 for r in records if r["model"] == m and r["label"] == 0)
        n_anom = sum(1 for r in records if r["model"] == m and r["label"] == 1)
        fp_all = sum(1 for r in records
                     if r["model"] == m and r["label"] == 0
                     and r["threshold"] is not None and r["score"] > r["threshold"])
        fn_all = sum(1 for r in records
                     if r["model"] == m and r["label"] == 1
                     and r["threshold"] is not None and r["score"] <= r["threshold"])
        summary_out[m] = {
            "num_normal": n_norm,
            "num_anomaly": n_anom,
            "false_positive": fp_all,
            "false_negative": fn_all,
            "false_positive_rate": round(fp_all / n_norm, 4) if n_norm else None,
            "false_negative_rate": round(fn_all / n_anom, 4) if n_anom else None,
        }

    (OUT_DIR / "failure_summary.json").write_text(
        json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 失败案例统计（F1 最优阈值下）=====")
    for m, s in summary_out.items():
        print(f"{m}: 误检 {s['false_positive']}/{s['num_normal']} "
              f"({s['false_positive_rate']}), "
              f"漏检 {s['false_negative']}/{s['num_anomaly']} "
              f"({s['false_negative_rate']})")
    print(f"\n输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
