"""
批量实验脚本: 多类别 x 多模型, 汇总结果为 CSV / Markdown

用法:
    py scripts/run_all.py                          # 使用配置中的类别与模型
    py scripts/run_all.py --categories bottle screw
    py scripts/run_all.py --models patchcore
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train import resolve_path, run_single


def df_to_md(df):
    """DataFrame 转 Markdown 表格（避免依赖 tabulate）"""
    lines = [
        "| " + " | ".join(str(c) for c in df.columns) + " |",
        "|" + "---|" * len(df.columns),
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="批量异常检测实验")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    categories = args.categories or cfg.get("categories") or ["bottle"]
    models = args.models or cfg.get("models") or ["patchcore", "padim"]

    rows = []
    t_start = time.time()
    for model_name in models:
        for category in categories:
            try:
                rows.append(run_single(model_name, category, cfg))
            except Exception as e:  # 单个类别失败不中断整体
                print(f"[错误] {model_name}/{category}: {e}")
                rows.append({"model": model_name, "category": category, "error": str(e)})

    df = pd.DataFrame(rows)
    out_dir = resolve_path(cfg["output"]["save_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    metric_cols = [c for c in ["image_auroc", "pixel_auroc", "aupro", "f1_max"] if c in df.columns]
    display_cols = ["model", "category"] + metric_cols
    if metric_cols:
        print("\n===== 实验结果汇总 =====")
        print(df[display_cols].to_string(index=False))

        ok = df.dropna(subset=metric_cols)
        if len(ok):
            print("\n===== 各模型平均 =====")
            mean_df = ok.groupby("model")[metric_cols].mean().round(4)
            print(mean_df.to_string())

            md = [
                "# 实验结果汇总",
                "",
                "## 各模型平均",
                "",
                df_to_md(mean_df.reset_index()),
                "",
                "## 明细",
                "",
                df_to_md(ok[display_cols].round(4)),
                "",
            ]
            (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n总耗时: {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"结果已保存: {csv_path}")


if __name__ == "__main__":
    main()
