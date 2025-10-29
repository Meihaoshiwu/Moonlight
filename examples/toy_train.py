import os
import math
import torch
import gc
import datetime
from loguru import logger
from contextlib import contextmanager
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm

# === 自定义存储路径 ===
MODEL_CACHE = "/home/ma-user/sfs_turbo/sudetong/models"  # 模型和tokenizer缓存路径
DATASET_CACHE = "/home/ma-user/sfs_turbo/sudetong/datasets"  # 数据集缓存路径
TOKENIZED_CACHE = "/home/ma-user/sfs_turbo/sudetong/tokenized_data"  # 分词后数据缓存路径

# 创建目录
os.makedirs(MODEL_CACHE, exist_ok=True)
os.makedirs(DATASET_CACHE, exist_ok=True)
os.makedirs(TOKENIZED_CACHE, exist_ok=True)

def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        # === 分词后数据缓存路径 ===
        self.cache_file = os.path.join(TOKENIZED_CACHE, f"{self.dataset_name}.bin")
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(self.cache_file):
            logger.info(f"Loading tokenized data from {self.cache_file}")
            self.tokens = torch.load(self.cache_file, weights_only=True)
        else:
            logger.info(f"Tokenizing texts and saving to {self.cache_file}")
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, self.cache_file)

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

def step_default(G, steps):
    return zeropower_via_newtonschulz5(G, steps)

def process_block(blocked_matrix, steps):
    if torch.norm(blocked_matrix) > 1e-7:
        orthogonalized_block = zeropower_via_newtonschulz5(blocked_matrix, steps)
        return orthogonalized_block
    else:
        return blocked_matrix

def step_column_block(G, steps):
    """
    将矩阵的列分成4块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G) # 用于保存结果
    cols = G.shape[1]  # 总列数
    block_size = cols // 4  # 每块的列数
    
    # 处理前3个完整的块
    for i in range(3):
        start_col = i * block_size
        end_col = (i + 1) * block_size
        column_block = G[:, start_col:end_col] # 取所有行，[start_col,end_col)列

        result[:, start_col:end_col] = process_block(column_block, steps)
    
    # 处理最后一块（可能包含剩余的列）
    start_col = 3 * block_size
    last_block = G[:, start_col:]
    result[:, start_col:] = process_block(last_block, steps)
    
    return result

def step_row_block(G, steps):
    """
    将矩阵的行分成4块，每块独立进行正交化处理
    """
    result = torch.zeros_like(G)
    rows = G.shape[0]  # 总行数
    block_size = rows // 4  # 每块的行数
    
    # 处理前3个完整的块
    for i in range(3):
        start_row = i * block_size
        end_row = (i + 1) * block_size
        row_block = G[start_row:end_row, :]
        result[start_row:end_row, :] = process_block(row_block, steps)
    
    # 处理最后一块（可能包含剩余的行）
    start_row = 3 * block_size
    last_block = G[start_row:, :]
    result[start_row:, :] = process_block(last_block, steps)
    
    return result

def step_quadrant_block(G, steps):
    """
    将矩阵分成2x2的分成四块
    """
    result = torch.zeros_like(G)
    
    rowidx = G.shape[0]//2
    colidx = G.shape[1]//2

    G11 = G[:rowidx, :colidx]
    result[:rowidx, :colidx] = process_block(G11, steps)
    G12 = G[:rowidx, colidx:]
    result[:rowidx, colidx:] = process_block(G12, steps)
    G21 = G[rowidx:, :colidx]
    result[rowidx:, :colidx] = process_block(G21, steps)
    G22 = G[rowidx:, colidx:]
    result[rowidx:, colidx:] = process_block(G22, steps)
    
    return result

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
        step_func:callable,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
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

        self.step_func = step_func
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
            # import pdb; pdb.set_trace()
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
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = self.step_func(g, steps=group["ns_steps"])

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


def get_model_and_dataloader(model_name, dataset_name, hidden_size, max_position_embeddings=2048, max_length=512, batch_size=32):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
        "openwebtext": "Skylion007/openwebtext",
        "wikitext-103": "wikitext",  # 约550MB
    }
    print(f"dataset_name = {dataset_name}, name2path[dataset_name] = {name2path[dataset_name]}")
    train_dataset = load_dataset(
        name2path[dataset_name], 
        cache_dir=DATASET_CACHE,  # 指定数据集缓存路径
    )
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", 
            cache_dir=MODEL_CACHE,  # 指定模型文件缓存路径
        )
    elif model_name == "qwen_small":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", 
            cache_dir=MODEL_CACHE,  # 指定模型文件缓存路径
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=4864,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=16,
            num_hidden_layers=12,
            num_key_value_heads=16,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    elif model_name == "qwen_small":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=hidden_size,
            initializer_range=0.02,
            intermediate_size=hidden_size*4,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=12,
            model_type="qwen2",
            num_attention_heads=8,
            num_hidden_layers=8,
            num_key_value_heads=8,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )
        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def get_optimizer(step_func:callable, optimizer_name, model, lr=1e-3, wd=0.1):
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
            step_func,
            lr=lr,
            wd=wd,
            muon_params=muon_params,
            adamw_params=adamw_params,
        )
    else:
        assert 0, "optimizer not supported"

# Python参数顺序规则,没有默认值参数要放在前
# 没有默认值的：位置参数  有默认值的：关键字参数
class ExperimentConfig():
    def __init__(
        self,
        step_func_name,
        step_func:callable,
        log_file_path,
        model_name="qwen",
        dataset_name="openwebtext",
        optimizer_name="muon",
        batch_size=32,
        hidden_size=512,
        max_position_embeddings=2048,
        max_length=512,
        loss_threshold=1.0,
        max_epochs=5,
        lr=1e-3,
        wd=0.1, #AdamW使用
    ):
        self.step_func_name = step_func_name
        self.step_func=step_func
        self.log_file_path = log_file_path
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.optimizer_name = optimizer_name
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.max_length = max_length
        self.loss_threshold = loss_threshold
        self.max_epochs = max_epochs
        self.lr = lr
        self.wd = wd
    
    def __repr__(self):
        """便于打印配置信息"""
        return (f"ExperimentConfig(step_func_name={self.step_func_name}, "
                f"model_name={self.model_name}, dataset_name={self.dataset_name}, "
                f"hidden_size={self.hidden_size}, loss_threshold={self.loss_threshold}, "
                f"max_epochs={self.max_epochs}, lr={self.lr}, wd={self.wd})")       

class ExperimentResources:
    """实验资源容器"""
    def __init__(
            self,
            model,
            train_loader,
            optimizer,
            device,
            lr_scheduler):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = lr_scheduler

# 实验函数配置
STEP_MAP = {
    "step_func_default": {"loss_threshold": 1.0, "step_func": step_default},
    "step_func_column_block": {"loss_threshold": 1.0, "step_func": step_column_block}, 
    "step_func_row_block": {"loss_threshold": 1.0, "step_func": step_row_block},
    "step_func_quadrant_block": {"loss_threshold": 1.0, "step_func": step_quadrant_block}
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@contextmanager
def experiment_manager(experiment_config: ExperimentConfig):
    """管理GPU、数据集、模型资源，管理日志打印"""
    step_func_name = experiment_config.step_func_name
    optimizer_name = experiment_config.optimizer_name
    loss_threshold = experiment_config.loss_threshold
    log_file_path = experiment_config.log_file_path
    lr = experiment_config.lr
    model_name = experiment_config.model_name
    dataset_name = experiment_config.dataset_name
    max_epochs = experiment_config.max_epochs
    max_position_embeddings=experiment_config.max_position_embeddings
    max_length = experiment_config.max_length
  
    # 控制日志范围 logger.add新建一个sink，后续的info都会打印在这里
    timestamp = get_timestamp()
    sink_id = logger.add(f"{log_file_path}/{timestamp}_train_{step_func_name}_{dataset_name}_{model_name}_{optimizer_name}_lr{lr}.log", mode="w")
    # 初始化所有资源
    model, train_loader = get_model_and_dataloader(
        model_name, dataset_name, experiment_config.hidden_size, max_position_embeddings=max_position_embeddings, max_length=max_length
    )

    total_params = count_parameters(model)
    logger.info(f"🚀 开始实验: {optimizer_name}_{step_func_name}")
    logger.info(f"目标损失阈值: {loss_threshold}, 最大epoch数: {max_epochs}, 总参数量: {total_params}"
                "max_position_embeddings: {max_position_embeddings}, max_length:{max_length}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = get_optimizer(
        STEP_MAP.get(step_func_name).get("step_func"), optimizer_name, model, lr=lr, wd=experiment_config.wd
    )
    
    num_training_steps = len(train_loader) * max_epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
        num_training_steps=num_training_steps,
        num_cycles=0.5,
    )
    
    # 封装资源
    resources = ExperimentResources(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        lr_scheduler=lr_scheduler
    )
    
    try:
        logger.info("✅ 资源初始化完成")
        yield resources  # 在这里交出资源控制权
        
    finally:
        # 清理资源（无论正常还是异常都会执行）
        logger.info("🧹 清理实验资源...")
        del resources.model
        del resources.train_loader
        del resources.optimizer
        del resources.lr_scheduler
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ 资源清理完成")
        logger.remove(sink_id)

def run_experiment(experiment_config: ExperimentConfig):
    """运行单个实验，直到损失低于阈值或达到最大epoch数"""
    with experiment_manager(experiment_config) as resources:
        model = resources.model
        train_loader = resources.train_loader
        optimizer = resources.optimizer
        device = resources.device
        lr_scheduler = resources.lr_scheduler
        step_func_name = experiment_config.step_func_name
        model.train()
        losses = []
        
        # 添加token计数器
        total_tokens_trained = 0
        batch_size = experiment_config.batch_size
        max_length = experiment_config.max_length
        
        for epoch in range(experiment_config.max_epochs):
            epoch_losses = []
            
            # 创建epoch进度条
            epoch_pbar = tqdm(
                total=len(train_loader),
                desc=f"Epoch {epoch+1}/{experiment_config.max_epochs} - {step_func_name}",
                unit="batch",
                ncols=100
            )
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch = batch.to(device)
                input_ids = batch
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                
                # 计算当前batch的token数并累加
                tokens_this_batch = batch_size * max_length
                total_tokens_trained += tokens_this_batch
                
                # 更新进度条描述
                epoch_pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'tokens': f'{total_tokens_trained:,}',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
                epoch_pbar.update(1)
                
                # 记录详细日志 - 改为token数对应loss
                if step % 100 == 0:
                    logger.info(
                        f"StepFunc: {step_func_name} Epoch: {epoch} Step: {step} Tokens: {total_tokens_trained} "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f} Loss: {current_loss:.4f}"
                    )
            
            epoch_pbar.close()
            
            # 计算epoch平均损失
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_epoch_loss)
            logger.info(f"📊 {step_func_name} - Epoch {epoch} 平均损失: {avg_epoch_loss:.4f}, 累计token数: {total_tokens_trained}")
            
            # 检查是否达到停止条件
            if avg_epoch_loss < experiment_config.loss_threshold:
                logger.info(f"🎯 {step_func_name} 已达到目标损失 {avg_epoch_loss:.4f} < {experiment_config.loss_threshold}, 停止训练")
                break
        final_loss = losses[-1] if losses else float('inf')
        return final_loss, losses

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--step_func_name", type=str, default="default")
    parser.add_argument("--model", type=str, default="qwen_small")
    parser.add_argument("--optimizer", type=str, default="muon")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--dataset", type=str, default="openwebtext-100k")
    parser.add_argument("--hidden_size", type=int, default=512) # 一个词使用多长的向量来表示（词向量维数）
    parser.add_argument("--max_position_embeddings", type=int, default=2048) # 最长给多少个词编码
    parser.add_argument("--max_length", type=int, default=256) # 每一批样本读入多少个token（即词向量）
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--loss_threshold", type=float, default=3.0)
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()

    timestamp = get_timestamp()
    base_log_path = "/home/ma-user/sfs_turbo/sudetong/results/logs/MuonBlockMatrix"
    experiment_dir = f"{base_log_path}/experiment_{timestamp}"

    experiment_config = ExperimentConfig(
        step_func_name="default",
        step_func=step_default,
        log_file_path=experiment_dir,
        model_name=args.model,
        dataset_name=args.dataset,
        optimizer_name=args.optimizer,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        loss_threshold=args.loss_threshold,
        max_epochs=args.max_epochs,
        lr=args.lr,
        wd=args.wd
    )

    print(f"🧪 开始自动化实验序列")
    print(f"使用模型缓存: {MODEL_CACHE}")
    print(f"使用数据集缓存: {DATASET_CACHE}")
    print(f"使用分词缓存: {TOKENIZED_CACHE}")
    
    results = {}
    for i, (func_name, config_dict) in enumerate(STEP_MAP.items()):
        # 运行实验
        experiment_config.step_func_name = func_name
        experiment_config.step_func = config_dict.get("step_func")
        final_loss, all_losses = run_experiment(experiment_config)
        
        # 保存结果
        results[func_name] = {
            'final_loss': final_loss,
            'all_losses': all_losses,
            'converged': final_loss < experiment_config.loss_threshold
        }
        
        # 为下一个实验等待一下，确保资源释放
        import time
        time.sleep(5)
    
    for exp_name, result in results.items():
        status = "✅ 收敛" if result['converged'] else "❌ 未收敛"
        print(f"{exp_name}: 最终损失 = {result['final_loss']:.4f} {status}")

if __name__ == "__main__":
    main()