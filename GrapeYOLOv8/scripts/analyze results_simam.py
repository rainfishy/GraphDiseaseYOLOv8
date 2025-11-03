"""
实验结果分析和对比脚本
生成论文所需的对比表格和图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_training_results():
    """分析训练结果并生成对比报告"""

    print("=" * 70)
    print("📊 YOLOv8n vs YOLOv8n+SimAM 实验结果分析")
    print("=" * 70)

    # 定义路径
    baseline_path = '../runs/baseline_yolov8n/weights/best.pt'
    simam_path = '../runs/train_simam/weights/best.pt'
    data_yaml = '../data_augmented/grape_augmented.yaml'
    output_dir = '../analysis_results'

    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print("\n📦 加载模型...")
    baseline = YOLO(baseline_path)
    simam = YOLO(simam_path)

    # 在测试集上验证
    print("\n🔍 在测试集上验证...")
    baseline_results = baseline.val(data=data_yaml, split='test')
    simam_results = simam.val(data=data_yaml, split='test')

    # 提取指标
    metrics_data = {
        '模型': ['YOLOv8n (Baseline)', 'YOLOv8n + SimAM'],
        'mAP@0.5': [
            baseline_results.box.map50,
            simam_results.box.map50
        ],
        'mAP@0.5:0.95': [
            baseline_results.box.map,
            simam_results.box.map
        ],
        'Precision': [
            baseline_results.box.mp,
            simam_results.box.mp
        ],
        'Recall': [
            baseline_results.box.mr,
            simam_results.box.mr
        ],
        'F1-Score': [
            2 * (baseline_results.box.mp * baseline_results.box.mr) /
            (baseline_results.box.mp + baseline_results.box.mr),
            2 * (simam_results.box.mp * simam_results.box.mr) /
            (simam_results.box.mp + simam_results.box.mr)
        ]
    }

    df = pd.DataFrame(metrics_data)

    # 计算提升
    improvements = {}
    for col in ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']:
        baseline_val = df.loc[0, col]
        simam_val = df.loc[1, col]
        improvement = ((simam_val - baseline_val) / baseline_val) * 100
        improvements[col] = improvement

    # 打印结果
    print("\n" + "=" * 70)
    print("📈 整体性能对比")
    print("=" * 70)
    print(df.to_string(index=False))

    print("\n" + "=" * 70)
    print("📊 性能提升")
    print("=" * 70)
    for metric, improvement in improvements.items():
        symbol = "📈" if improvement > 0 else "📉"
        print(f"{symbol} {metric}: {improvement:+.2f}%")

    # 保存CSV
    csv_path = os.path.join(output_dir, 'overall_comparison.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 整体对比已保存: {csv_path}")

    # 生成对比图表
    generate_comparison_plots(df, improvements, output_dir)

    # 各类别对比
    analyze_per_class_performance(baseline_results, simam_results, output_dir)

    # 生成LaTeX表格
    generate_latex_table(df, improvements, output_dir)

    print("\n" + "=" * 70)
    print("✅ 分析完成！结果已保存到:", output_dir)
    print("=" * 70)


def generate_comparison_plots(df, improvements, output_dir):
    """生成对比图表"""

    print("\n📊 生成对比图表...")

    # 1. 性能对比柱状图
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['mAP@0.5', 'mAP@0.5:0.95', 'Precision', 'Recall', 'F1-Score']
    x = range(len(metrics))
    width = 0.35

    baseline_vals = [df.loc[0, m] for m in metrics]
    simam_vals = [df.loc[1, m] for m in metrics]

    bars1 = ax.bar([i - width / 2 for i in x], baseline_vals, width,
                   label='YOLOv8n (Baseline)', color='skyblue')
    bars2 = ax.bar([i + width / 2 for i in x], simam_vals, width,
                   label='YOLOv8n + SimAM', color='orange')

    ax.set_xlabel('指标', fontsize=12)
    ax.set_ylabel('数值', fontsize=12)
    ax.set_title('YOLOv8n vs YOLOv8n+SimAM 性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300)
    print("  ✅ 性能对比图已保存")
    plt.close()

    # 2. 提升百分比图
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['green' if v > 0 else 'red' for v in improvements.values()]
    bars = ax.barh(list(improvements.keys()), list(improvements.values()), color=colors)

    ax.set_xlabel('提升百分比 (%)', fontsize=12)
    ax.set_title('SimAM改进带来的性能提升', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, improvements.values())):
        ax.text(val, i, f' {val:+.2f}%', va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improvement_percentage.png'), dpi=300)
    print("  ✅ 提升百分比图已保存")
    plt.close()


def analyze_per_class_performance(baseline_results, simam_results, output_dir):
    """各类别性能分析"""

    print("\n📊 分析各类别性能...")

    # 类别名称
    class_names = ['black_rot', 'blight', 'black_measles', 'Healthy']

    # 提取各类别的mAP@0.5
    baseline_maps = baseline_results.box.maps  # 各类别mAP
    simam_maps = simam_results.box.maps

    # 创建DataFrame
    per_class_data = {
        '类别': class_names,
        'Baseline mAP@0.5': baseline_maps.tolist(),
        'SimAM mAP@0.5': simam_maps.tolist()
    }

    df_class = pd.DataFrame(per_class_data)
    df_class['提升 (%)'] = ((df_class['SimAM mAP@0.5'] - df_class['Baseline mAP@0.5']) /
                            df_class['Baseline mAP@0.5'] * 100)

    print("\n" + "=" * 70)
    print("📊 各类别性能对比")
    print("=" * 70)
    print(df_class.to_string(index=False))

    # 保存CSV
    csv_path = os.path.join(output_dir, 'per_class_comparison.csv')
    df_class.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 各类别对比已保存: {csv_path}")

    # 生成各类别对比图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(class_names))
    width = 0.35

    bars1 = ax.bar([i - width / 2 for i in x], df_class['Baseline mAP@0.5'], width,
                   label='Baseline', color='skyblue')
    bars2 = ax.bar([i + width / 2 for i in x], df_class['SimAM mAP@0.5'], width,
                   label='SimAM', color='orange')

    ax.set_xlabel('类别', fontsize=12)
    ax.set_ylabel('mAP@0.5', fontsize=12)
    ax.set_title('各类别检测性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加提升百分比标签
    for i, improvement in enumerate(df_class['提升 (%)']):
        y_pos = max(df_class.loc[i, 'Baseline mAP@0.5'],
                    df_class.loc[i, 'SimAM mAP@0.5']) + 0.02
        color = 'green' if improvement > 0 else 'red'
        ax.text(i, y_pos, f'{improvement:+.1f}%', ha='center',
                fontsize=10, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_comparison.png'), dpi=300)
    print("  ✅ 各类别对比图已保存")
    plt.close()


def generate_latex_table(df, improvements, output_dir):
    """生成LaTeX格式表格（用于论文）"""

    print("\n📝 生成LaTeX表格...")

    latex_content = r"""\begin{table}[htbp]
\centering
\caption{YOLOv8n与YOLOv8n+SimAM性能对比}
\label{tab:performance_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{模型} & \textbf{mAP@0.5} & \textbf{mAP@0.5:0.95} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} & \textbf{参数量} \\
\midrule
"""

    for idx, row in df.iterrows():
        model = row['模型']
        latex_content += f"{model} & "
        latex_content += f"{row['mAP@0.5']:.4f} & "
        latex_content += f"{row['mAP@0.5:0.95']:.4f} & "
        latex_content += f"{row['Precision']:.4f} & "
        latex_content += f"{row['Recall']:.4f} & "
        latex_content += f"{row['F1-Score']:.4f} & "
        latex_content += "3.0M \\\\\n"

    latex_content += r"""\midrule
\textbf{提升} & """

    latex_content += f"\\textbf{{{improvements['mAP@0.5']:+.2f}\%}} & "
    latex_content += f"\\textbf{{{improvements['mAP@0.5:0.95']:+.2f}\%}} & "
    latex_content += f"{improvements['Precision']:+.2f}\% & "
    latex_content += f"\\textbf{{{improvements['Recall']:+.2f}\%}} & "
    latex_content += f"{improvements['F1-Score']:+.2f}\% & "
    latex_content += "0 \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    # 保存LaTeX代码
    latex_path = os.path.join(output_dir, 'performance_table.tex')
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"  ✅ LaTeX表格已保存: {latex_path}")


if __name__ == "__main__":
    analyze_training_results()