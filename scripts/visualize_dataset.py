"""
MVTec AD 数据集可视化脚本
生成数据探索的各种图表
"""
import os
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / 'data' / 'raw' / 'mvtec_anomaly_detection'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'visualization'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 加载统计数据
stats_file = PROJECT_ROOT / 'results' / 'dataset_stats.json'
with open(stats_file, 'r', encoding='utf-8') as f:
    stats = json.load(f)

categories = sorted(stats.keys())

# ============ 1. 样本对比展示 ============
def visualize_samples():
    print("生成样本对比展示...")
    num_cats = 5
    selected_cats = categories[:num_cats]
    
    fig = plt.figure(figsize=(20, 4 * num_cats))
    
    for i, cat in enumerate(selected_cats):
        cat_path = DATA_ROOT / cat
        
        # 训练正常
        train_imgs = list((cat_path / 'train' / 'good').glob('*.png'))
        if train_imgs:
            ax = fig.add_subplot(num_cats, 5, i * 5 + 1)
            ax.imshow(Image.open(train_imgs[0]))
            ax.set_title(f'{cat}\nTrain Normal', fontsize=10)
            ax.axis('off')
        
        # 测试正常
        test_imgs = list((cat_path / 'test' / 'good').glob('*.png'))
        if test_imgs:
            ax = fig.add_subplot(num_cats, 5, i * 5 + 2)
            ax.imshow(Image.open(test_imgs[0]))
            ax.set_title('Test Normal', fontsize=10)
            ax.axis('off')
        
        # 测试异常
        test_dir = cat_path / 'test'
        defect_types = [d for d in os.listdir(test_dir) if d != 'good' and (test_dir / d).is_dir()]
        if defect_types:
            defect_imgs = list((test_dir / defect_types[0]).glob('*.png'))
            if defect_imgs:
                ax = fig.add_subplot(num_cats, 5, i * 5 + 3)
                ax.imshow(Image.open(defect_imgs[0]))
                ax.set_title(f'Test Anomaly\n{defect_types[0]}', fontsize=10)
                ax.axis('off')
        
        # Ground Truth
        gt_dir = cat_path / 'ground_truth'
        if gt_dir.exists() and defect_types:
            gt_imgs = list((gt_dir / defect_types[0]).glob('*_mask.png'))
            if gt_imgs:
                ax = fig.add_subplot(num_cats, 5, i * 5 + 4)
                ax.imshow(Image.open(gt_imgs[0]), cmap='gray')
                ax.set_title('Ground Truth', fontsize=10)
                ax.axis('off')
        
        # 叠加图
        if defect_types and gt_dir.exists():
            defect_imgs = list((test_dir / defect_types[0]).glob('*.png'))
            gt_imgs = list((gt_dir / defect_types[0]).glob('*_mask.png'))
            if defect_imgs and gt_imgs:
                ax = fig.add_subplot(num_cats, 5, i * 5 + 5)
                img = np.array(Image.open(defect_imgs[0]).convert('RGB'))
                mask = np.array(Image.open(gt_imgs[0]).convert('L'))
                overlay = img.copy()
                overlay[:, :, 0] = np.clip(img[:, :, 0] + mask * 0.5, 0, 255)
                ax.imshow(overlay)
                ax.set_title('Overlay', fontsize=10)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '01_sample_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '01_sample_comparison.png'}")

# ============ 2. 图像尺寸分布 ============
def visualize_size_distribution():
    print("生成图像尺寸分布...")
    sizes = {}
    for cat in categories:
        cat_path = DATA_ROOT / cat
        train_imgs = list((cat_path / 'train' / 'good').glob('*.png'))
        if train_imgs:
            img = Image.open(train_imgs[0])
            sizes[cat] = img.size
    
    widths = [s[0] for s in sizes.values()]
    heights = [s[1] for s in sizes.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(categories, widths, color='#667eea', edgecolor='white')
    axes[0].set_title('Image Width by Category', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Width (pixels)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].axhline(y=np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(categories, heights, color='#764ba2', edgecolor='white')
    axes[1].set_title('Image Height by Category', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Height (pixels)')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '02_size_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '02_size_distribution.png'}")

# ============ 3. 颜色分布 ============
def visualize_color_distribution():
    print("生成颜色分布...")
    selected_cats = ['bottle', 'cable', 'capsule', 'leather', 'wood']
    
    fig, axes = plt.subplots(len(selected_cats), 3, figsize=(15, 3 * len(selected_cats)))
    
    for i, cat in enumerate(selected_cats):
        cat_path = DATA_ROOT / cat
        train_imgs = list((cat_path / 'train' / 'good').glob('*.png'))[:5]
        
        all_pixels = []
        for img_path in train_imgs:
            img = Image.open(img_path).resize((256, 256))
            pixels = np.array(img).reshape(-1, 3)
            all_pixels.append(pixels)
        all_pixels = np.vstack(all_pixels)
        
        colors = ['red', 'green', 'blue']
        for j, (color_name, color) in enumerate(zip(['R', 'G', 'B'], colors)):
            axes[i, j].hist(all_pixels[:, j], bins=50, color=color, alpha=0.7, edgecolor='white')
            axes[i, j].set_title(f'{cat} - {color_name} Channel', fontsize=10)
            axes[i, j].set_xlabel('Pixel Value')
            axes[i, j].set_ylabel('Count')
            axes[i, j].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '03_color_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '03_color_distribution.png'}")

# ============ 4. 缺陷类型分布 ============
def visualize_defect_distribution():
    print("生成缺陷类型分布...")
    defect_counts = {cat: len(stats[cat]['defect_types']) for cat in categories}
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 柱状图
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = axes[0].barh(categories, [defect_counts[c] for c in categories], color=colors)
    axes[0].set_title('Number of Defect Types per Category', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Number of Defect Types')
    axes[0].grid(axis='x', alpha=0.3)
    for bar, count in zip(bars, [defect_counts[c] for c in categories]):
        axes[0].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, str(count), 
                    va='center', fontsize=9)
    
    # 饼图
    all_defects = sum(defect_counts.values())
    axes[1].pie(defect_counts.values(), labels=defect_counts.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[1].set_title(f'Defect Type Distribution (Total: {all_defects})', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '04_defect_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '04_defect_distribution.png'}")

# ============ 5. 训练/测试样本对比 ============
def visualize_train_test_comparison():
    print("生成训练/测试样本对比...")
    train_counts = [stats[c]['train'] for c in categories]
    test_normal = [stats[c]['test_good'] for c in categories]
    test_anomaly = [stats[c]['test_anomaly'] for c in categories]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    x = np.arange(len(categories))
    width = 0.25
    
    axes[0].bar(x - width, train_counts, width, label='Train', color='#667eea')
    axes[0].bar(x, test_normal, width, label='Test Normal', color='#48bb78')
    axes[0].bar(x + width, test_anomaly, width, label='Test Anomaly', color='#f56565')
    axes[0].set_title('Sample Distribution by Category', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Number of Images')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # 堆叠柱状图
    axes[1].bar(categories, train_counts, label='Train', color='#667eea')
    axes[1].bar(categories, test_normal, bottom=train_counts, label='Test Normal', color='#48bb78')
    axes[1].bar(categories, test_anomaly, bottom=[t+n for t,n in zip(train_counts, test_normal)], 
               label='Test Anomaly', color='#f56565')
    axes[1].set_title('Total Samples per Category (Stacked)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Number of Images')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '05_train_test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '05_train_test_comparison.png'}")

# ============ 6. 亮度分布 ============
def visualize_brightness_distribution():
    print("生成亮度分布...")
    selected_cats = ['bottle', 'cable', 'capsule', 'leather', 'wood']
    
    fig, axes = plt.subplots(len(selected_cats), 2, figsize=(12, 3 * len(selected_cats)))
    
    for i, cat in enumerate(selected_cats):
        cat_path = DATA_ROOT / cat
        
        # 正常图像亮度
        normal_imgs = list((cat_path / 'train' / 'good').glob('*.png'))[:10]
        normal_brightness = []
        for img_path in normal_imgs:
            img = Image.open(img_path).convert('L')
            normal_brightness.append(np.mean(np.array(img)))
        
        # 异常图像亮度
        test_dir = cat_path / 'test'
        defect_types = [d for d in os.listdir(test_dir) if d != 'good' and (test_dir / d).is_dir()]
        anomaly_brightness = []
        for dt in defect_types[:3]:
            anomaly_imgs = list((test_dir / dt).glob('*.png'))[:5]
            for img_path in anomaly_imgs:
                img = Image.open(img_path).convert('L')
                anomaly_brightness.append(np.mean(np.array(img)))
        
        axes[i, 0].hist(normal_brightness, bins=20, color='#48bb78', alpha=0.7, edgecolor='white')
        axes[i, 0].set_title(f'{cat} - Normal Brightness', fontsize=10)
        axes[i, 0].set_xlabel('Mean Brightness')
        axes[i, 0].set_ylabel('Count')
        axes[i, 0].grid(alpha=0.3)
        
        axes[i, 1].hist(anomaly_brightness, bins=20, color='#f56565', alpha=0.7, edgecolor='white')
        axes[i, 1].set_title(f'{cat} - Anomaly Brightness', fontsize=10)
        axes[i, 1].set_xlabel('Mean Brightness')
        axes[i, 1].set_ylabel('Count')
        axes[i, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '06_brightness_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '06_brightness_distribution.png'}")

# ============ 7. 异常样本展示 ============
def visualize_anomaly_samples():
    print("生成异常样本展示...")
    selected_cats = ['bottle', 'cable', 'capsule', 'carpet', 'leather']
    
    fig = plt.figure(figsize=(20, 12))
    
    for i, cat in enumerate(selected_cats):
        cat_path = DATA_ROOT / cat
        test_dir = cat_path / 'test'
        gt_dir = cat_path / 'ground_truth'
        
        defect_types = [d for d in os.listdir(test_dir) if d != 'good' and (test_dir / d).is_dir()]
        
        for j, dt in enumerate(defect_types[:4]):
            idx = i * 4 + j + 1
            if idx > 20:
                break
            
            defect_imgs = list((test_dir / dt).glob('*.png'))
            if defect_imgs:
                ax = fig.add_subplot(5, 4, idx)
                img = Image.open(defect_imgs[0])
                ax.imshow(img)
                ax.set_title(f'{cat} - {dt}', fontsize=9)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '07_anomaly_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '07_anomaly_samples.png'}")

# ============ 8. 数据集总览 ============
def visualize_dataset_overview():
    print("生成数据集总览...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 各类别样本总数
    totals = [stats[c]['train'] + stats[c]['test_good'] + stats[c]['test_anomaly'] for c in categories]
    axes[0, 0].bar(categories, totals, color='#667eea', edgecolor='white')
    axes[0, 0].set_title('Total Samples per Category', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 正常 vs 异常比例
    total_normal = sum(stats[c]['train'] + stats[c]['test_good'] for c in categories)
    total_anomaly = sum(stats[c]['test_anomaly'] for c in categories)
    axes[0, 1].pie([total_normal, total_anomaly], labels=['Normal', 'Anomaly'], 
                  autopct='%1.1f%%', colors=['#48bb78', '#f56565'], startangle=90)
    axes[0, 1].set_title('Normal vs Anomaly Ratio', fontsize=11, fontweight='bold')
    
    # 各类别异常占比
    anomaly_ratios = [stats[c]['test_anomaly'] / (stats[c]['test_good'] + stats[c]['test_anomaly']) * 100 
                     for c in categories]
    axes[1, 0].bar(categories, anomaly_ratios, color='#f56565', edgecolor='white')
    axes[1, 0].set_title('Anomaly Ratio in Test Set (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Anomaly Ratio (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 缺陷类型数量分布
    defect_counts = [len(stats[c]['defect_types']) for c in categories]
    axes[1, 1].hist(defect_counts, bins=range(1, 10), color='#764ba2', edgecolor='white', align='left')
    axes[1, 1].set_title('Distribution of Defect Types Count', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Defect Types')
    axes[1, 1].set_ylabel('Number of Categories')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '08_dataset_overview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {RESULTS_DIR / '08_dataset_overview.png'}")

# ============ 主函数 ============
if __name__ == '__main__':
    print("=" * 60)
    print("MVTec AD 数据集可视化")
    print("=" * 60)
    
    visualize_samples()
    visualize_size_distribution()
    visualize_color_distribution()
    visualize_defect_distribution()
    visualize_train_test_comparison()
    visualize_brightness_distribution()
    visualize_anomaly_samples()
    visualize_dataset_overview()
    
    print("\n" + "=" * 60)
    print(f"所有可视化已保存至: {RESULTS_DIR}")
    print("=" * 60)
