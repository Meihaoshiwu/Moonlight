import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

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

def plot_training_comparison(log_directory, output_path="training_comparison.png"):
    """绘制训练对比图"""
    # 获取所有日志文件
    log_files = glob.glob(os.path.join(log_directory, "*.log"))
    
    if not log_files:
        print(f"在目录 {log_directory} 中未找到任何日志文件")
        return
    
    plt.figure(figsize=(12, 8))
    
    all_data = []
    
    for log_file in log_files:
        # 解析日志文件
        exp_name, tokens, losses = parse_log_file(log_file)
        
        if tokens and losses:
            # 绘制曲线
            plt.plot(tokens, losses, marker='o', markersize=4, linewidth=2, label=exp_name)
            
            # 收集数据用于表格
            all_data.append({
                '实验名称': exp_name,
                '初始损失': losses[0],
                '最终损失': losses[-1],
                '总训练Token数': tokens[-1]
            })
    
    if not all_data:
        print("没有找到有效的训练数据")
        return
    
    plt.xlabel('训练Token数', fontsize=14)
    plt.ylabel('损失', fontsize=14)
    plt.title('优化器性能对比', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 设置x轴为科学计数法（如果数值很大）
    if max([data['总训练Token数'] for data in all_data]) > 1000000:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"图表已保存至: {output_path}")
    
    # 生成详细数据表格
    create_detailed_table(all_data)

def create_detailed_table(data):
    """创建详细的数据表格"""
    # 计算改善百分比
    for item in data:
        improvement = ((item['初始损失'] - item['最终损失']) / item['初始损失']) * 100
        item['改善百分比'] = f"{improvement:.1f}%"
        item['初始损失'] = f"{item['初始损失']:.4f}"
        item['最终损失'] = f"{item['最终损失']:.4f}"
        item['总训练Token数'] = f"{item['总训练Token数']:,}"
    
    # 创建DataFrame并排序
    df = pd.DataFrame(data)
    df = df.sort_values('最终损失')
    
    print("\n详细实验结果对比:")
    print("=" * 80)
    print(df.to_string(index=False))
    
    # 保存表格到CSV
    df.to_csv("training_results.csv", index=False, encoding='utf-8-sig')
    print(f"\n详细数据已保存至: training_results.csv")
    
    return df

if __name__ == "__main__":
    # 设置你的日志目录路径
    log_directory = "/home/ma-user/sfs_turbo/sudetong/LOG/MounBlockMatrix"
    outputpath = "/home/ma-user/sfs_turbo/sudetong/LOG/pics/training_comparison.png"
    
    # 生成对比图
    plot_training_comparison(log_directory, outputpath)