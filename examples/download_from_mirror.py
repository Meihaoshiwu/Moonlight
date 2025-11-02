import os
# 修改为镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# === 自定义存储路径 ===
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR")
# === 基于环境变量的存储路径 ===
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
DATASET_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets") 
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")

from huggingface_hub import snapshot_download

local_dir = 'path_to_down'

snapshot_download(
  repo_id="Skylion007/openwebtext",
  repo_type="dataset",
  local_dir=DATASET_CACHE,
  # allow_patterns=['Charades/', 'DiDeMo/', 'HiREST/', 'activitynet/', 'coin/', 'querYD/', 'qvhighlights/', 'youcook2/'],
  local_dir_use_symlinks=False,
  resume_download=True,
  # proxies={"https": "http://localhost:7890"}, # clash default port
  max_workers=8
)