import os
import math
import torch
import time, datetime
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 同步CUDA操作，便于调试

# === 自定义存储路径 ===
MUON_BLOCK_MATRIX_EXPERIMENT_DIR = os.getenv("MUON_BLOCK_MATRIX_EXPERIMENT_DIR", "./experiments")
# === 基于环境变量的存储路径 ===
MODEL_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Models")
DATASET_PATH = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Datasets")
DATASET_CACHE = os.path.join(DATASET_PATH, "cache")
DATASET_DOWNLOAD = os.path.join(DATASET_PATH, "download")
RESULTS_BASE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "Results")
TOKENIZED_CACHE = os.path.join(MUON_BLOCK_MATRIX_EXPERIMENT_DIR, "TokenizedData")

class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=1024):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        cache_dir = TOKENIZED_CACHE
        cache_file = os.path.join(cache_dir, f"{self.dataset_name}.bin")
        if os.path.exists(cache_file):
            self.tokens = torch.load(cache_file, weights_only=True)
            logger.info(f"Loaded cached tokens from {cache_file}")
        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, cache_file)
            logger.info(f"Saved tokens to {cache_file}")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        cpu_offloading:bool,
        lr=1e-3,
        wd=0.1, # 保持与论文一致
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.9), # 保持与论文一致
        adamw_eps=1e-8,
    ):

        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        self.cpu_offloading = cpu_offloading
        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    if self.cpu_offloading:
                        # CPU offloading版本：初始化到CPU
                        state["momentum_buffer"] = torch.zeros_like(g, device='cpu')
                    else:
                        # 普通版本：初始化到GPU
                        state["momentum_buffer"] = torch.zeros_like(g)
                
                # 获取动量缓冲区
                buf = state["momentum_buffer"]
                if self.cpu_offloading:
                    # CPU offloading版本：需要将buffer移动到GPU进行计算
                    buf = buf.to(g.device)
                
                # 更新动量
                buf.mul_(momentum).add_(g)
                
                if group["nesterov"]:
                    g_update = g.add(buf, alpha=momentum)
                else:
                    g_update = buf
                
                # 正交化更新
                u = zeropower_via_newtonschulz5(g_update, steps=group["ns_steps"])
                # 更新动量缓冲区状态
                if self.cpu_offloading:
                    # CPU offloading版本：将buffer移回CPU
                    state["momentum_buffer"] = buf.cpu()
                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters())

def get_gpu_memory():
    """获取GPU显存使用情况"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # 转换为GB
    return 0

def setup_logger(model_name, cpu_offloading, batch_size, log_dir):
    """设置日志文件"""
    # 根据CPU offloading状态和batch_size添加后缀
    suffix = f"{'cpu_offload' if cpu_offloading else 'gpu_only'}_bs{batch_size}"
    log_file = os.path.join(log_dir, f"{model_name}_{suffix}.log")
    
    # 配置logger
    logger.remove()  # 移除默认的handler
    sink_id = logger.add(log_file, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info(f"日志文件: {log_file}")
    logger.info(f"CPU Offloading: {cpu_offloading}")
    logger.info(f"Batch Size: {batch_size}")
    
    return sink_id

def _create_model(model_name):
    if model_name == "llama_130m":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,  # LLaMA通常使用1作为BOS
            eos_token_id=2,  # LLaMA通常使用2作为EOS
            hidden_act="silu",
            hidden_size=792,
            initializer_range=0.02,
            intermediate_size=2048,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=12,
            num_hidden_layers=16,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,  # RoPE基础频率
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,  # LLaMA-2 tokenizer词汇表大小
        )
        model = LlamaForCausalLM(config)

    elif model_name == "llama_350m":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=1024,
            initializer_range=0.02,
            intermediate_size=2560,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=16,
            num_hidden_layers=30,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,
        )
        model = LlamaForCausalLM(config)

    elif model_name == "llama_1.1b":
        config = LlamaConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=2048,
            initializer_range=0.02,
            intermediate_size=5632,  # FFN size
            max_position_embeddings=1024,
            model_type="llama",
            num_attention_heads=32,
            num_hidden_layers=24,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            vocab_size=32000,
        )
        model = LlamaForCausalLM(config)
    else:
        logger.error(f"不支持的模型: {model_name}")
        return None
        
    return model

def get_model_and_dataloader(model_name, dataset_name, batch_size):
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
            logger.info(f"✅ 数据集 '{dataset_name}' 加载成功")
            logger.info(f"   训练集大小: {len(dataset['train'])} 条样本")
        except Exception as e:
            logger.error(f"❌ 处理数据集 '{dataset_name}' 时出错: {e}")
            return None, None
    else:
        logger.error(f"❌ 未知数据集: {dataset_name}")
        return None, None

    if model_name.startswith("llama"):
        # 使用正确的LLaMA tokenizer
        logger.info(f"模型: {model_name}获取tokenzier")
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            "huggyllama/llama-7b",
            trust_remote_code=True
        )
    elif model_name.startswith("qwen"):
        logger.info(f"模型: {model_name}获取tokenzier")
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        logger.error(f"❌ 未知模型: {model_name}")
        return None, None
    
    train_dataset = MoonDataset(dataset_name, dataset, tokenizer)
    
    # 根据batch_size设置数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = _create_model(model_name)
    
    # 关键：打印词汇表大小信息
    logger.info(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"Tokenizer special tokens: pad={tokenizer.pad_token_id}, eos={tokenizer.eos_token_id}, unk={tokenizer.unk_token_id}")
        
    return model, train_loader

def get_optimizer(optimizer_name, cpu_offloading:bool, model, lr=1e-3, wd=0.1):
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95)
        )
    elif optimizer_name == "muon":
        muon_params = [
            p
            for name, p in model.named_parameters()
            if p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
        ]
        adamw_params = [
            p
            for name, p in model.named_parameters()
            if not (
                p.ndim >= 2 and "embed_tokens" not in name and "lm_head" not in name
            )
        ]

        return Muon(
            cpu_offloading=cpu_offloading,
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        logger.error(f"不支持的优化器: {optimizer_name}")
        return None

def train_model(model_name, cpu_offloading, batch_size, num_training_steps, log_dir, dataset_name="openwebtext-100k", optimizer_name="muon", lr=1e-3, wd=0.1):
    """训练单个模型"""
    
    # 设置日志
    sink_id = setup_logger(model_name, cpu_offloading, batch_size, log_dir)
    
    logger.info(f"开始训练模型: {model_name}")
    logger.info(f"优化器: {optimizer_name}")
    logger.info(f"CPU Offloading: {cpu_offloading}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"学习率: {lr}, 权重衰减: {wd}")

    # 设置训练参数
    if batch_size == 1024:
        # 对于1024 batch_size，使用梯度累积
        accumulation_steps = 256  # 4 * 256 = 1024
        actual_batch_size = 4  # 每个step实际处理的样本数
        logger.info(f"开启梯度累积，梯度累积步数: {accumulation_steps}")
        logger.info(f"实际每个step处理的样本数: {actual_batch_size}")
        logger.info(f"总训练步数: {num_training_steps}")
    else:
        # 对于batch_size=1，直接训练
        accumulation_steps = 1
        actual_batch_size = batch_size
        logger.info(f"不开启梯度累积，每次训练{batch_size}个样本")
        logger.info(f"总训练步数: {num_training_steps}")
    
    # 获取模型和数据加载器
    model, train_loader = get_model_and_dataloader(model_name, dataset_name, actual_batch_size)
    if model is None or train_loader is None:
        logger.error(f"获取模型或数据加载器失败")
        return None, None, None, None
    
    # 计算参数量
    total_params = count_parameters(model)
    logger.info(f"模型参数量: {total_params:,}")
    
    # 获取优化器
    optimizer = get_optimizer(optimizer_name, cpu_offloading, model, lr=lr, wd=wd)
    if optimizer is None:
        return None, None, None, None
    
    # 添加余弦退火调度器
    num_warmup_steps = 2  # 10%的warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    logger.info(f"使用余弦退火调度器: warmup步数={num_warmup_steps}, 总步数={num_training_steps}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    model.to(device)
    
    # 记录初始显存
    initial_memory = get_gpu_memory()
    logger.info(f"初始显存占用: {initial_memory:.2f} GB")
    
    model.train()
    
    # 训练参数
    total_time = 0
    optimizer_times = []
    step_times = []
    memory_usage = []
    
    logger.info(f"开始训练，共 {num_training_steps} 个等效更新步骤")
    
    for step in range(num_training_steps):
        step_start_time = time.time()
        
        try:
            # 梯度累积
            optimizer.zero_grad()
            accumulated_loss = 0
            
            for accum_step in range(accumulation_steps):
                # 获取一个batch
                batch = next(iter(train_loader))
                batch = batch.to(device)
                
                # 前向传播
                input_ids = batch
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                
                # 反向传播，梯度累积
                (loss / accumulation_steps).backward()
                accumulated_loss += loss.item()
            
            # 更新参数
            optimizer_start_time = time.time()
            optimizer.step()
            cur_optimizer_time = time.time() - optimizer_start_time
            optimizer_times.append(cur_optimizer_time)
            scheduler.step()  # 更新学习率
            
            # 计算步骤时间
            step_time = time.time() - step_start_time
            total_time += step_time
            step_times.append(step_time)
            
            # 获取当前显存和学习率
            current_memory = get_gpu_memory()
            current_lr = scheduler.get_last_lr()[0]
            
            # 记录最后10步的显存使用
            if step >= num_training_steps - 10:
                memory_usage.append(current_memory)
            
            # 记录日志
            logger.info(
                f"Step: {step+1}/{num_training_steps} | "
                f"Loss: {accumulated_loss/accumulation_steps:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Step Time: {step_time:.3f}s | "
                f"Optimizer Time: {cur_optimizer_time:.3f}s | "
                f"GPU Memory: {current_memory:.2f}GB"
            )
            
        except Exception as e:
            logger.error(f"训练步骤 {step+1} 失败: {e}")
            break
    
    # 计算平均时间
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0
    
    # 计算最后10步的平均显存和平均时间
    last_10_avg_memory = sum(memory_usage) / len(memory_usage) if memory_usage else 0
    last_10_avg_time = sum(step_times[-10:]) / len(step_times[-10:]) if len(step_times) >= 10 else avg_step_time
    last_10_avg_optimizer_time = sum(optimizer_times[-10:]) / len(optimizer_times[-10:]) if len(optimizer_times) >= 10 else -1
    
    # 记录总结信息
    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成总结 - {model_name} (CPU Offloading: {cpu_offloading}, BS: {batch_size}):")
    logger.info(f"{'='*60}")
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"总训练时间: {total_time:.2f} 秒")
    logger.info(f"平均每步时间: {avg_step_time:.3f} 秒")
    logger.info(f"最后10步平均显存: {last_10_avg_memory:.2f} GB")
    logger.info(f"最后10步平均时间: {last_10_avg_time:.3f} 秒")
    logger.info(f"最后10步纯优化器平均用时: {last_10_avg_optimizer_time:.3f} 秒")
    logger.info(f"最终显存占用: {get_gpu_memory():.2f} GB")
    logger.info(f"日志文件保存在: {log_dir}")
    logger.info(f"{'='*60}")

    logger.remove(sink_id)
    
    return avg_step_time, total_time, total_params, last_10_avg_memory

TRAINING_STEPS=12
def main():
    """主函数，分别训练三个模型的两个版本（CPU offloading和普通版本）和两种batch_size"""
    
    # 三个要训练的模型
    models_to_train = ["llama_130m", "llama_350m", "llama_1.1b"]
    # 两种batch_size
    batch_sizes = [1024, 1]
    dataset_name = "openwebtext-100k"
    optimizer_name = "muon"  # 使用Muon优化器进行对比
    lr = 1e-3
    wd = 0.1
    
    # 存储结果用于对比
    results = {}
    
    # 创建带时间戳的日志目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(RESULTS_BASE, "Logs", f"batchsize_experiment_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    logger.info("开始批量训练三个LLaMA模型，对比CPU Offloading和Batch Size效果")
    logger.info(f"日志目录: {log_dir}")
    
    total_experiments = len(models_to_train) * len(batch_sizes) * 2  # 3模型 * 2batch_size * 2CPU设置
    current_experiment = 0

    for model_name in models_to_train:
        results[model_name] = {}
        
        for batch_size in batch_sizes:
            results[model_name][f"bs_{batch_size}"] = {}
            
            # 分别运行CPU offloading版本和普通版本
            for cpu_offloading in [True, False]:
                current_experiment += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"实验 {current_experiment}/{total_experiments}")
                logger.info(f"开始训练模型: {model_name} | CPU Offloading: {cpu_offloading} | Batch Size: {batch_size}")
                logger.info(f"{'='*80}")
                
                try:
                    avg_step_time, total_time, total_params, last_10_avg_memory = train_model(
                        model_name=model_name,
                        cpu_offloading=cpu_offloading,
                        batch_size=batch_size,
                        num_training_steps=TRAINING_STEPS,
                        log_dir=log_dir,
                        dataset_name=dataset_name,
                        optimizer_name=optimizer_name,
                        lr=lr,
                        wd=wd
                    )
                    
                    if avg_step_time is not None:
                        # 存储结果
                        cpu_key = "cpu_offload" if cpu_offloading else "gpu_only"
                        results[model_name][f"bs_{batch_size}"][cpu_key] = {
                            "avg_step_time": avg_step_time,
                            "total_time": total_time,
                            "total_params": total_params,
                            "last_10_avg_memory": last_10_avg_memory,
                            "last_10_avg_time": sum([avg_step_time] * 10) / 10  # 简化计算
                        }
                
                except Exception as e:
                    logger.error(f"训练模型 {model_name} (CPU Offloading: {cpu_offloading}, BS: {batch_size}) 时发生错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 添加实验间延迟
                time.sleep(2)
    
    # 输出对比结果
    logger.info(f"\n{'='*80}")
    logger.info("CPU Offloading 和 Batch Size 对比结果总结")
    logger.info(f"{'='*80}")
    
    for model_name in models_to_train:
        if model_name in results:
            logger.info(f"\n模型: {model_name}")
            
            for batch_size in batch_sizes:
                bs_key = f"bs_{batch_size}"
                if bs_key in results[model_name]:
                    logger.info(f"\n  Batch Size: {batch_size}")
                    
                    cpu_offload_result = results[model_name][bs_key].get("cpu_offload")
                    gpu_only_result = results[model_name][bs_key].get("gpu_only")
                    
                    if cpu_offload_result and gpu_only_result:
                        logger.info(f"    CPU Offloading 最后10步平均显存: {cpu_offload_result['last_10_avg_memory']:.2f}GB")
                        logger.info(f"    GPU Only 最后10步平均显存: {gpu_only_result['last_10_avg_memory']:.2f}GB")
                        logger.info(f"    CPU Offloading 最后10步平均时间: {cpu_offload_result['last_10_avg_time']:.3f}s")
                        logger.info(f"    GPU Only 最后10步平均时间: {gpu_only_result['last_10_avg_time']:.3f}s")
                        logger.info(f"    显存差异: {cpu_offload_result['last_10_avg_memory'] - gpu_only_result['last_10_avg_memory']:.2f}GB")
                        logger.info(f"    时间差异: {cpu_offload_result['last_10_avg_time'] - gpu_only_result['last_10_avg_time']:.3f}s")
    
    logger.info("所有模型训练完成！")


if __name__ == "__main__":
    main()