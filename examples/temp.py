import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def get_model_and_dataloader(model_name, dataset_name, hidden_size, max_position_embeddings=2048, max_length=512, batch_size=32, rank=0, world_size=1):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
        "openwebtext": "Skylion007/openwebtext",
        "wikitext-103": "wikitext",  # 约550MB
    }
    print(f"dataset_name = {dataset_name}, name2path[dataset_name] = {name2path[dataset_name]}")

    # 修改：从本地txt文件加载数据集
    if dataset_name == "openwebtext-local_txt":
        train_dataset = load_dataset("text", data_files=f"{OPENWEBTEXT_EXTRACTED}/*.txt", streaming=True,
            cache_dir=DATASET_CACHE,  # 指定数据集缓存路径
        )
    else:
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
    
    # DDP修改：使用DistributedSampler
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=(sampler is None)  # 单GPU时使用shuffle
    )

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
    return model, train_loader, sampler

@contextmanager
def experiment_manager(experiment_config: ExperimentConfig, rank=0, world_size=1):
    """管理GPU、数据集、模型资源，管理日志打印 - DDP兼容版本"""
    step_func_name = experiment_config.step_func_name
    optimizer_name = experiment_config.optimizer_name
    loss_threshold = experiment_config.loss_threshold
    log_file_path = experiment_config.log_file_path
    lr = experiment_config.lr
    model_name = experiment_config.model_name
    dataset_name = experiment_config.dataset_name
    max_epochs = experiment_config.max_epochs
    max_position_embeddings = experiment_config.max_position_embeddings
    max_length = experiment_config.max_length
  
    # DDP修改：每个进程有独立的日志文件
    timestamp = get_timestamp()
    logger.remove()
    
    # 只在主进程记录详细日志，其他进程只记录错误
    if rank == 0:
        sink_id = logger.add(
            f"{log_file_path}/{timestamp}_train_{step_func_name}_{dataset_name}_{model_name}_{optimizer_name}_lr{lr}.log", 
            mode="w", level="INFO"
        )
    else:
        # 其他进程只记录错误到独立文件
        sink_id = logger.add(
            f"{log_file_path}/{timestamp}_rank{rank}_{step_func_name}.log", 
            mode="w", level="ERROR"
        )
    
    # 初始化所有资源
    model, train_loader, sampler = get_model_and_dataloader(
        model_name, dataset_name, experiment_config.hidden_size, 
        max_position_embeddings=max_position_embeddings, 
        max_length=max_length,
        batch_size=experiment_config.batch_size,
        rank=rank,
        world_size=world_size
    )

    total_params = count_parameters(model)
    
    # 只在主进程显示启动信息
    if rank == 0:
        logger.info(f"🚀 开始实验: {optimizer_name}_{step_func_name}")
        logger.info(f"目标损失阈值: {loss_threshold}, 最大epoch数: {max_epochs}, 总参数量: {total_params}")
        logger.info(f"max_position_embeddings: {max_position_embeddings}, max_length:{max_length}")
        logger.info(f"使用DDP训练，检测到 {world_size} 个GPU")
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # DDP包装
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 获取优化器时需要注意：DDP包装后需要通过module访问原始模型
    original_model = model.module if world_size > 1 else model
    optimizer = get_optimizer(
        STEP_MAP.get(step_func_name).get("step_func"), 
        optimizer_name, 
        original_model, 
        lr=lr, 
        wd=experiment_config.wd
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
        lr_scheduler=lr_scheduler,
        sampler=sampler,
        rank=rank,
        world_size=world_size
    )
    
    try:
        if rank == 0:
            logger.info("✅ 资源初始化完成")
        yield resources  # 在这里交出资源控制权
        
    finally:
        # 清理资源（无论正常还是异常都会执行）
        if rank == 0:
            logger.info("🧹 清理实验资源...")
        
        # DDP清理
        if world_size > 1:
            cleanup_ddp()
        
        del resources.model
        del resources.train_loader
        del resources.optimizer
        del resources.lr_scheduler
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if rank == 0:
            logger.info("✅ 资源清理完成")
        logger.remove(sink_id)

def train_worker(rank, world_size, experiment_config):
    """DDP训练工作进程"""
    with experiment_manager(experiment_config, rank, world_size) as resources:
        model = resources.model
        train_loader = resources.train_loader
        optimizer = resources.optimizer
        device = resources.device
        lr_scheduler = resources.lr_scheduler
        sampler = resources.sampler
        step_func_name = experiment_config.step_func_name
        
        model.train()
        losses = []
        total_tokens_trained = 0
        batch_size = experiment_config.batch_size
        max_length = experiment_config.max_length
        
        for epoch in range(experiment_config.max_epochs):
            # DDP重要：设置epoch以便shuffle正常工作
            if sampler:
                sampler.set_epoch(epoch)
                
            epoch_losses = []
            
            # 只在主进程显示进度条
            if rank == 0:
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
                
                # DDP重要：如果使用了DDP，loss已经是所有GPU的平均值
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                
                # 计算当前batch的token数并累加
                tokens_this_batch = batch_size * max_length
                total_tokens_trained += tokens_this_batch
                
                # 只在主进程更新进度条和日志
                if rank == 0:
                    epoch_pbar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'tokens': f'{total_tokens_trained:,}',
                        'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                    })
                    epoch_pbar.update(1)
                    
                    if step % 100 == 0:
                        logger.info(
                            f"StepFunc: {step_func_name} Epoch: {epoch} Step: {step} Tokens: {total_tokens_trained} "
                            f"LR: {optimizer.param_groups[0]['lr']:.6f} Loss: {current_loss:.4f}"
                        )
            
            if rank == 0:
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

def run_experiment(experiment_config: ExperimentConfig):
    """运行单个实验，支持DDP"""
    world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # 多GPU使用DDP
        logger.info(f"使用DDP训练，检测到 {world_size} 个GPU")
        mp.spawn(
            train_worker,
            args=(world_size, experiment_config),
            nprocs=world_size,
            join=True
        )
        # 注意：在DDP模式下，返回值处理需要额外逻辑
        return None, None
    else:
        # 单GPU直接调用训练函数
        return train_worker(0, 1, experiment_config)