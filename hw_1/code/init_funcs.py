import os
import random

import numpy as np
import torch
import utils
import wandb
from data import get_dataloader
# from data_copy import get_dataloader

from omegaconf import OmegaConf
from torch import distributed as dist
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817


def create_task(cfg, rank):
    if rank == 0:
        wandb.init(
            project=cfg.project_name,
            name=cfg.exp_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        return  wandb.run


def gather_and_init(config):  # noqa: WPS210
    init_seeds(config)
    rank, model = init_ddp_model(config)
    use_amp, scaler = init_mixed_precision(config)
    outdir = create_save_folder(config, rank)
    task = create_task(config, rank)

    train_loader = get_dataloader(config, split_type='train', task=task, rank=rank)
    num_batches = len(train_loader)
    criterion, optimizer, scheduler = utils.get_training_parameters(config, model, num_batches)
    mixer = utils.get_mixer(config)

    train_options = {
        'train_loader': train_loader,
        'val_loader': get_dataloader(config, split_type='val', task=task, rank=rank),
        'loss': criterion,
        'optimizer': optimizer,
        'use_amp': use_amp,
        'scaler': scaler,
        'rank': rank,
        'scheduler': scheduler,
        'outdir': outdir,
        'mixer': mixer,
    }

    return model, train_options, task


def create_save_folder(config, rank):
    outdir = os.path.join(config.outdir, config.exp_name)
    if (rank == 0) and (not os.path.exists(outdir)):
        os.makedirs(outdir)
    return outdir


def init_seeds(config):
    seed = config.dataset.seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def init_mixed_precision(config):
    use_amp = False
    if hasattr(config.train, 'use_amp'):
        if config.train.use_amp:
            use_amp = True

    if use_amp:
        scaler = GradScaler()
    else:
        scaler = None

    return use_amp, scaler


def init_ddp_model(config):
    dist.init_process_group(backend='nccl', init_method='env://')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    net = utils.load_model(config, is_train=True).cuda()
    #sync_bn_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net, rank)
    ddp_net = DDP(net, device_ids=[rank])

    return rank, ddp_net
