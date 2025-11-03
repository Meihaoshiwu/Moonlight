import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import numpy as np
import argparse

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用黑体，如果黑体不存在就用DejaVu Sans
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def parse_log_file(log_file_path):
    """解析日志文件，提取训练步数和损失值"""
    tokens = []
    losses = []
    exp_name = os.path.basename(log_file_path).replace('.log', '')
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配包含token数和损失的行
                if "Tokens:" in line and "Loss:" in line:
                    # 提取token数
                    token_match = re.search(r"Tokens: (\d+)", line)
                    # 提取损失值
                    loss_match = re.search(r"Loss: ([\d.]+)", line)
                    
                    if token_match and loss_match:
                        tokens.append(int(token_match.group(1)))
                        losses.append(float(loss_match.group(1)))
    except Exception as e:
        print(f"解析日志文件 {log_file_path} 时出错: {e}")
    
    return exp_name, tokens, losses

def plot_training_comparison(log_directory, output_path, exclude_keywords=None):
    """绘制训练对比图
    
    Args:
        log_directory: 日志文件目录（必要参数）
        output_path: 输出图像路径，包括目录和文件名（必要参数）
        exclude_keywords: 要排除的关键字列表，包含这些关键字的文件不会被处理
    """
    if exclude_keywords is None:
        exclude_keywords = ['rank']  # 默认排除包含rank的文件
    
    # 获取所有日志文件
    log_files = glob.glob(os.path.join(log_directory, "*.log"))
    
    if not log_files:
        print(f"在目录 {log_directory} 中未找到任何日志文件")
        return
    
    # 过滤掉包含排除关键字的文件
    filtered_log_files = []
    for log_file in log_files:
        filename = os.path.basename(log_file).lower()
        should_exclude = any(keyword.lower() in filename for keyword in exclude_keywords)
        
        if not should_exclude:
            filtered_log_files.append(log_file)
        else:
            print(f"跳过文件（包含排除关键字）: {os.path.basename(log_file)}")
    
    if not filtered_log_files:
        print(f"在过滤后，目录 {log_directory} 中没有可用的日志文件")
        return
    
    print(f"处理 {len(filtered_log_files)} 个日志文件:")
    for log_file in filtered_log_files:
        print(f"  - {os.path.basename(log_file)}")
    
    # 创建更大的图像
    plt.figure(figsize=(14, 10))
    
    # 定义丰富的颜色和线型组合
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    all_data = []
    
    for i, log_file in enumerate(filtered_log_files):
        # 解析日志文件
        exp_name, tokens, losses = parse_log_file(log_file)
        
        if tokens and losses:
            # 选择颜色、线型和标记
            color = colors[i % len(colors)]
            line_style = line_styles[(i // len(colors)) % len(line_styles)]
            marker = markers[i % len(markers)]
            
            # 绘制曲线，减少标记密度
            step = max(1, len(tokens) // 20)  # 最多显示20个标记点
            plt.plot(tokens, losses, 
                    color=color, 
                    linestyle=line_style, 
                    marker=marker, 
                    markersize=6, 
                    markevery=step,
                    linewidth=2.5, 
                    label=exp_name,
                    alpha=0.8)
            
            # 收集数据用于表格
            all_data.append({
                '实验名称': exp_name,
                '初始损失': losses[0],
                '最终损失': losses[-1],
                '总训练Token数': tokens[-1]
            })
        else:
            print(f"警告: 文件 {exp_name} 中没有找到有效数据")
    
    if not all_data:
        print("没有找到有效的训练数据")
        return
    
    plt.xlabel('训练Token数', fontsize=16, fontweight='bold')
    plt.ylabel('损失', fontsize=16, fontweight='bold')
    plt.title('优化器性能对比', fontsize=18, fontweight='bold', pad=20)
    
    # 改进图例显示
    plt.legend(fontsize=12, loc='best', framealpha=0.9, 
               fancybox=True, shadow=True, ncol=1 if len(filtered_log_files) <= 5 else 2)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置坐标轴格式
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置x轴为科学计数法（如果数值很大）
    if max([data['总训练Token数'] for data in all_data]) > 1000000:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    
    # 根据损失值范围调整y轴
    all_losses = []
    for data in all_data:
        exp_name = data['实验名称']
        for log_file in filtered_log_files:
            if exp_name in log_file:
                _, tokens, losses = parse_log_file(log_file)
                all_losses.extend(losses)
                break
    
    if all_losses:
        y_min = max(0, min(all_losses) * 0.8)  # 留一些边距
        y_max = min(max(all_losses) * 1.2, max(all_losses) + 0.1)  # 防止损失值过大
        plt.ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"图表已保存至: {output_path}")
    
    # 生成详细数据表格
    create_detailed_table(all_data, output_dir)

def create_detailed_table(data, output_dir):
    """创建详细的数据表格
    
    Args:
        data: 要展示的数据
        output_dir: 输出目录，用于保存CSV文件
    """
    # 计算改善百分比
    for item in data:
        improvement = ((item['初始损失'] - item['最终损失']) / item['初始损失']) * 100
        item['改善百分比'] = f"{improvement:.1f}%"
        item['初始损失'] = f"{item['初始损失']:.4f}"
        item['最终损失'] = f"{item['最终损失']:.4f}"
        item['总训练Token数'] = f"{item['总训练Token数']:,}"
    
    # 创建DataFrame并排序
    df = pd.DataFrame(data)
    # 按最终损失排序，损失最小的排在最前面
    df = df.sort_values('最终损失')
    
    print("\n详细实验结果对比:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存表格到CSV
    csv_path = os.path.join(output_dir, "training_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存至: {csv_path}")
    
    return df

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='生成训练对比图')
    parser.add_argument('--log_dir', required=True, 
                       help='日志文件目录（必要参数）')
    parser.add_argument('--output_path', required=True,
                       help='输出图像路径，包括目录和文件名（必要参数）')
    parser.add_argument('--exclude_keywords', nargs='+', default=['rank'],
                       help='要排除的关键字列表，默认为["rank"]')
    
    args = parser.parse_args()
    
    # 检查日志目录是否存在
    if not os.path.exists(args.log_dir):
        print(f"错误: 日志目录不存在: {args.log_dir}")
        return
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"创建输出目录: {output_dir}")
    
    # 生成对比图
    plot_training_comparison(args.log_dir, args.output_path, args.exclude_keywords)

if __name__ == "__main__":
    main()