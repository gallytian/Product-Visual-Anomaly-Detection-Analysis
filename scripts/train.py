"""
单类别实验脚本: 构建模型 + 测试评估 + 保存结果

用法:
    py scripts/train.py --model patchcore --category bottle
    py scripts/train.py --model padim --category bottle --config configs/default.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloader
from src.metrics import compute_metrics
from src.models.padim import PaDiM
from src.models.patchcore import PatchCore
from src.visualize import visualize_results


def resolve_path(p):
    """把相对路径解析到项目根目录"""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def build_model(model_name, cfg, device):
    """按配置构建模型"""
    mcfg = cfg["model"][model_name]
    if model_name == "patchcore":
        return PatchCore(
            layers=mcfg["layers"],
            coreset_ratio=mcfg["coreset_ratio"],
            n_neighbors=mcfg["n_neighbors"],
            max_bank_size=mcfg.get("max_bank_size", 40000),
            projection_dim=mcfg.get("projection_dim", 128),
            smooth_sigma=mcfg.get("smooth_sigma", 4.0),
            device=device,
            seed=cfg.get("seed", 42),
        )
    if model_name == "padim":
        return PaDiM(
            layers=mcfg["layers"],
            n_dims=mcfg["n_dims"],
            eps=mcfg.get("eps", 0.01),
            smooth_sigma=mcfg.get("smooth_sigma", 4.0),
            device=device,
            seed=cfg.get("seed", 42),
        )
    raise ValueError(f"未知模型: {model_name}")


def run_single(model_name, category, cfg, verbose=True):
    """运行单个模型在单个类别上的完整实验, 返回指标字典"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds_root = resolve_path(cfg["dataset"]["root"])
    resize = cfg["dataset"]["resize"]
    batch_size = cfg["dataset"]["batch_size"]
    num_workers = cfg["dataset"]["num_workers"]

    train_loader = get_dataloader(ds_root, category, "train", batch_size, num_workers, resize)
    test_loader = get_dataloader(ds_root, category, "test", batch_size, num_workers, resize)

    model = build_model(model_name, cfg, device)

    print(f"  构建记忆库/分布参数 ({len(train_loader.dataset)} 张训练图)...", flush=True)
    t0 = time.time()
    fit_info = model.fit(train_loader)
    fit_time = time.time() - t0
    print(f"  完成: {fit_info} ({fit_time:.0f}s)", flush=True)

    print(f"  测试集评估 ({len(test_loader.dataset)} 张)...", flush=True)
    scores, labels, maps, masks, paths = [], [], [], [], []
    t0 = time.time()
    done = 0
    for images, y, m, p in test_loader:
        s, amap = model.predict(images)
        scores.append(np.asarray(s))
        labels.append(np.asarray(y))
        maps.append(amap)
        masks.append(np.asarray(m))
        paths.extend(list(p))
        done += len(p)
        if done % 40 < len(p):
            print(f"    进度 {done}/{len(test_loader.dataset)}", flush=True)
    eval_time = time.time() - t0

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    maps = np.concatenate(maps)
    masks = np.concatenate(masks)

    metrics = compute_metrics(scores, labels, maps, masks)
    metrics.update({
        "model": model_name,
        "category": category,
        "fit_time_sec": round(fit_time, 1),
        "eval_time_sec": round(eval_time, 1),
        "num_test": int(len(scores)),
    })
    metrics.update({f"fit_{k}": v for k, v in fit_info.items()})

    out_dir = resolve_path(cfg["output"]["save_dir"]) / model_name / category
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    np.savez_compressed(
        out_dir / "scores.npz",
        scores=scores.astype(np.float32),
        labels=labels.astype(np.int64),
        maps=maps.astype(np.float16),
        paths=np.array(paths),
    )

    # 可视化: 取分数最高的异常样本 + 若干正常样本
    order = np.argsort(-scores)
    n_anom = cfg["output"].get("num_visualize", 6)
    sel_anom = [int(i) for i in order if labels[int(i)] == 1][:n_anom]
    sel_norm = [int(i) for i in order if labels[int(i)] == 0][: max(1, n_anom // 2)]
    sel = sel_anom + sel_norm
    if sel:
        visualize_results(
            [paths[i] for i in sel],
            maps[sel],
            masks[sel],
            labels[sel],
            out_dir / "visualizations",
            n_samples=len(sel),
        )

    if verbose:
        print(
            f"[{model_name}/{category}] image_auroc={metrics['image_auroc']:.4f} "
            f"pixel_auroc={metrics.get('pixel_auroc', float('nan')):.4f} "
            f"aupro={metrics.get('aupro', float('nan')):.4f} "
            f"f1_max={metrics['f1_max']:.4f} "
            f"fit={fit_time:.0f}s eval={eval_time:.0f}s"
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="单类别异常检测实验")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--model", type=str, default="patchcore", choices=["patchcore", "padim"])
    parser.add_argument("--category", type=str, default="bottle")
    args = parser.parse_args()

    with open(resolve_path(args.config), "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device} | 模型: {args.model} | 类别: {args.category}")
    run_single(args.model, args.category, cfg)


if __name__ == "__main__":
    main()
