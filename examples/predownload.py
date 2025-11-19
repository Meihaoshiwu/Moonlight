#!/usr/bin/env python3
import os
import argparse
from transformers import Qwen2Tokenizer, LlamaTokenizer

from datasets import load_dataset
from loguru import logger
# === 自定义存储路径 ===
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
# === 基于环境变量的存储路径 ===
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
DATASET_PATH = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets")
DATASET_CACHE = os.path.join(DATASET_PATH, "cache")
DATASET_DOWNLOAD = os.path.join(DATASET_PATH, "download")

# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def pre_download_resources(dataset_names, model_name):
    """预下载所有需要的资源"""
    
    # 确保目录存在
    os.makedirs(MODEL_CACHE, exist_ok=True)
    os.makedirs(DATASET_DOWNLOAD, exist_ok=True)
    
    logger.info("🔽 开始预下载资源...")
    
    # 下载Tokenizer
    _download_tokenizer(model_name)
    
    # 下载数据集
    if dataset_names:
        _download_datasets(dataset_names)
    
    logger.info("🎉 所有资源预下载完成！")
    logger.info(f"模型缓存: {MODEL_CACHE}")
    logger.info(f"数据集下载到: {DATASET_DOWNLOAD}")

def _download_tokenizer(model_name):
    """下载模型对应的tokenizer"""
    logger.info(f"下载Tokenizer: {model_name}")
    
    if model_name.startswith("qwen"):
        tokenizer_name = "Qwen/Qwen2.5-0.5B"
        tokenizer_class = Qwen2Tokenizer
    elif model_name.startswith("llama"):
        tokenizer_name = "huggyllama/llama-7b"
        tokenizer_class = LlamaTokenizer
    
    try:
        tokenizer_class.from_pretrained(
            tokenizer_name,
            cache_dir=MODEL_CACHE,
            local_files_only=True
        )
        logger.info("✅ Tokenizer已存在于缓存中")
    except (OSError, TypeError):
        tokenizer_class.from_pretrained(
            tokenizer_name,
            cache_dir=MODEL_CACHE
        )
        logger.info(f"✅ Tokenizer已下载到: {MODEL_CACHE}")

def _download_datasets(dataset_names):
    """下载指定的数据集"""
    for dataset_name in dataset_names:
        logger.info(f"下载数据集: {dataset_name}")
        
        # 数据集名称到路径的映射
        name2path = {
            "openwebtext-100k": "Elriggs/openwebtext-100k",
            "openwebtext": "Skylion007/openwebtext",
            "wikitext-103": "wikitext",
        }
        
        if dataset_name in name2path:
            try:
                dataset = load_dataset(
                    name2path[dataset_name],
                    cache_dir=DATASET_CACHE,
                )
                logger.info(f"✅ 数据集 '{dataset_name}' 本地缓存存在")
                logger.info(f"   训练集大小: {len(dataset['train'])} 条样本")
            except Exception as e:
                logger.error(f"❌ 处理数据集 '{dataset_name}' 时出错: {e}")
        else:
            logger.warning(f"⚠️  未知数据集: {dataset_name}，跳过下载")

def main():
    parser = argparse.ArgumentParser(description="预下载训练所需资源")
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["openwebtext-100k"],
        help="要下载的数据集名称，支持: openwebtext-100k, openwebtext, wikitext-103"
    )
    parser.add_argument(
        "--model_name",
    )
    
    args = parser.parse_args()
    
    pre_download_resources(
        dataset_names=args.datasets,
        model_name=args.model_name
    )

if __name__ == "__main__":
    main()