import argparse

import numpy as np
import torch
import random
from models.models import XProNet
from modules.dataloaders import R2DataLoader
from modules.loss import compute_loss
from modules.metrics import CaptionScorer
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from modules.utils import parse_agrs
import torch.distributed as dist
import os
from modules.logger import create_logger


def setup(world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = world_size


def main():
    # parse arguments
    args = parse_agrs()

    # DDP settings
    world_size = args.n_gpu

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size)
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    # torch.distributed.barrier()

    # fix random seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    save_dir = os.path.join(args.output, args.dataset_name, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = create_logger(output_dir=save_dir, dist_rank=args.local_rank, name=args.exp_name)

    # create tokenizer
    # create tokenizer
    if args.dataset_name == 'cxr_gnome':
        tokenizer = None
    else:
        tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, drop_last=True)

    if args.dataset_name == 'cxr_gnome':
        tokenizer = train_dataloader.dataset.tokenizer
    all_texts = tokenizer.all_texts

    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = XProNet(args, tokenizer)
    state_dict = torch.load(os.path.join(args.trained_model_path))['state_dict']
    model.load_state_dict(state_dict)
    # change this with your pretrained model path

    optimizer = build_optimizer(args, model)

    model = model.to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id], broadcast_buffers=False,
                                                      find_unused_parameters=True)

    model_without_ddp = model.module

    if dist.get_rank() == args.local_rank:
        logger.info(args)
        logger.info(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
        n_parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        logger.info(f"number of params: {n_parameters}")
        if hasattr(model_without_ddp, 'flops'):
            flops = model_without_ddp.flops()
            logger.info(f"number of GFLOPs: {flops / 1e9}")

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = CaptionScorer(all_texts)

    # build optimizer, learning rate scheduler
    lr_scheduler = build_lr_scheduler(args, optimizer)
    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, logger, train_dataloader,
                      val_dataloader,
                      test_dataloader)
    trainer.test()


if __name__ == '__main__':
    main()
