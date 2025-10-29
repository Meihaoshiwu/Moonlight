# extract_openwebtext.py
import os
import sys
import tarfile
import lzma
import re

# 添加原始 openwebtext.py 的路径到系统路径
sys.path.append('/path/to/your/dataset/directory')

# 假设你已经下载的数据在这个目录
DATA_DIR = "/home/ma-user/sfs_turbo/sudetong/datasets/download/subsets"
OUTPUT_DIR = "/home/ma-user/sfs_turbo/sudetong/datasets/openwebtext_extracted"

def extract_dataset():
    """手动解压 OpenWebText 数据集"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 找到所有的 tar 文件
    tar_files = [os.path.join(DATA_DIR, f"urlsf_subset{i:02d}.tar") for i in range(21)]
    
    total_files = 0
    for tar_path in tar_files:
        if not os.path.exists(tar_path):
            print(f"跳过不存在的文件: {tar_path}")
            continue
            
        print(f"处理文件: {tar_path}")
        
        # 解压 tar 文件
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if member.name.endswith('.xz'):
                    # 提取 xz 文件
                    xz_file = tar.extractfile(member)
                    if xz_file:
                        # 解压 xz 并读取内容
                        try:
                            with lzma.open(xz_file, 'rt', encoding='utf-8') as f:
                                text_content = f.read()
                            
                            # 清理文本（与原脚本相同的处理）
                            cleaned_text = re.sub("\n\n\n+", "\n\n", text_content).strip()
                            
                            # 保存为 txt 文件
                            output_filename = os.path.join(
                                OUTPUT_DIR, 
                                f"doc_{total_files:06d}.txt"
                            )
                            with open(output_filename, 'w', encoding='utf-8') as out_file:
                                out_file.write(cleaned_text)
                            
                            total_files += 1
                            if total_files % 1000 == 0:
                                print(f"已处理 {total_files} 个文档...")
                                
                        except Exception as e:
                            print(f"处理文件 {member.name} 时出错: {e}")
                            continue
    
    print(f"解压完成！共提取 {total_files} 个文档到 {OUTPUT_DIR}")

if __name__ == "__main__":
    extract_dataset()