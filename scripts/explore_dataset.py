import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'mvtec_anomaly_detection'

def analyze_dataset():
    print("=" * 80)
    print("MVTec AD 数据集分析")
    print("=" * 80)
    
    stats = defaultdict(lambda: {'train': 0, 'test_good': 0, 'test_anomaly': 0, 'defect_types': [], 'ground_truth_count': 0})
    categories = sorted([d for d in os.listdir(DATA_ROOT) if (DATA_ROOT / d).is_dir()])
    
    for category in categories:
        cat_path = DATA_ROOT / category
        train_good = cat_path / 'train' / 'good'
        if train_good.exists():
            stats[category]['train'] = len(list(train_good.glob('*.png')))
        
        test_dir = cat_path / 'test'
        if test_dir.exists():
            for defect_type in os.listdir(test_dir):
                defect_path = test_dir / defect_type
                if defect_path.is_dir():
                    count = len(list(defect_path.glob('*.png')))
                    if defect_type == 'good':
                        stats[category]['test_good'] = count
                    else:
                        stats[category]['test_anomaly'] += count
                        stats[category]['defect_types'].append(defect_type)
        
        gt_dir = cat_path / 'ground_truth'
        if gt_dir.exists():
            for defect_type in os.listdir(gt_dir):
                gt_path = gt_dir / defect_type
                if gt_path.is_dir():
                    stats[category]['ground_truth_count'] += len(list(gt_path.glob('*_mask.png')))
    
    print(f"\n{'类别':<15} {'训练':>6} {'测试正常':>8} {'测试异常':>8} {'GT掩码':>8} {'缺陷类型':>8}")
    print("-" * 80)
    
    total_train = total_test = total_gt = total_defects = 0
    for cat in sorted(stats.keys()):
        s = stats[cat]
        total_train += s['train']
        total_test += s['test_good'] + s['test_anomaly']
        total_gt += s['ground_truth_count']
        total_defects += len(s['defect_types'])
        print(f"{cat:<15} {s['train']:>6} {s['test_good']:>8} {s['test_anomaly']:>8} {s['ground_truth_count']:>8} {len(s['defect_types']):>8}")
    
    print("-" * 80)
    print(f"{'总计':<15} {total_train:>6} {total_test:>8} {total_gt:>8} {total_defects:>8}")
    print(f"\n共 {len(stats)} 个类别")
    return stats

def analyze_image_properties():
    print("\n分析图像属性...")
    sample_categories = ['bottle', 'cable', 'capsule']
    for cat_name in sample_categories:
        cat_path = DATA_ROOT / cat_name
        if not cat_path.is_dir():
            continue
        print(f"\n{cat_name}:")
        train_good = cat_path / 'train' / 'good'
        if train_good.exists():
            train_images = list(train_good.glob('*.png'))[:1]
            if train_images:
                img = Image.open(train_images[0])
                print(f"  训练图像尺寸: {img.size}, 模式: {img.mode}")
        gt_dir = cat_path / 'ground_truth'
        if gt_dir.exists():
            for defect_type in os.listdir(gt_dir):
                gt_path = gt_dir / defect_type
                if gt_path.is_dir():
                    gt_images = list(gt_path.glob('*_mask.png'))[:1]
                    if gt_images:
                        mask = Image.open(gt_images[0])
                        print(f"  掩码尺寸: {mask.size}, 模式: {mask.mode}")
                    break

def save_statistics(stats):
    output_path = PROJECT_ROOT / 'results' / 'dataset_stats.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_dict = {cat: dict(data) for cat, data in stats.items()}
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)
    print(f"\n统计信息已保存至: {output_path}")

if __name__ == '__main__':
    stats = analyze_dataset()
    analyze_image_properties()
    save_statistics(stats)
    print("\n" + "=" * 80)
    print("数据探索完成！")
    print("=" * 80)
