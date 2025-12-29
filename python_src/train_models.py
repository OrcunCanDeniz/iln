import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

# Datasets
from dataset.range_images_dataset import RangeImagesDataset
from dataset.samples_from_image_dataset import SamplesFromImageDataset
from dataset.dataset_utils import generate_dataset

# Models
from models.iln.iln import ILN
from models.liif_cvpr21.liif_lidar import LIIFLiDAR
from models.lsr_ras20.unet import UNet
from models.model_utils import generate_model

# Global rank variable for logging guards
RANK = 0

def is_valid_check_point():
    if check_point['model']['name'] != config['model']['name']:
        return False
    for key, value in check_point['model']['args'].items():
        if value != config['model']['args'][key]:
            return False
    if check_point['lidar_in'] != train_dataset.lidar_in:
        return False
    return True

def print_log(epoch, loss_sum, loss_avg, directory=None):
    # Only Rank 0 writes logs
    if RANK != 0:
        return

    log_msg = ('%03d %.4f %.4f' % (epoch, loss_sum, loss_avg))
    
    # Ensure directory exists
    os.makedirs(directory, exist_ok=True)
    
    with open(os.path.join(directory, 'training_loss_history.txt'), 'a') as f:
        f.write(log_msg + '\n')
    print(log_msg)

def save_check_point(epoch, period=10):
    # Only Rank 0 saves checkpoints
    if RANK != 0:
        return

    suffix = 'latest'
    for i in range(2):
        # Access .module because DDP wraps the model
        model_state = model.module.state_dict() if isinstance(model, (DDP, nn.DataParallel)) else model.state_dict()
        
        check_point_model_info = {'name': config['model']['name'],
                                  'args': config['model']['args'],
                                  'state_dict': model_state}
        check_point = {'epoch': epoch + 1,
                       'model': check_point_model_info,
                       'optimizer': optimizer.state_dict(),
                       'lr_scheduler': lr_scheduler.state_dict(),
                       'lidar_in': train_dataset.lidar_in}

        check_point_filename = os.path.join(model_directory, model_name + '_' + suffix + '.pth')
        torch.save(check_point, check_point_filename)
        
        if epoch % period == period-1:
            suffix = str(epoch + 1)
        else:
            break

def train_implicit_network():
    for epoch in range(epoch_start, epoch_end, 1):
        # Important for DDP: Shuffle data differently every epoch
        train_loader.sampler.set_epoch(epoch)
        
        loss_sum = 0.0
        
        # Only show tqdm on Rank 0 to avoid messy output
        iterator = tqdm(train_loader, leave=False, desc='train') if RANK == 0 else train_loader

        for input_range_images, input_queries, output_ranges in iterator:
            input_range_images = input_range_images.cuda(non_blocking=True)
            input_queries = input_queries.cuda(non_blocking=True)
            output_ranges = output_ranges.cuda(non_blocking=True)

            optimizer.zero_grad()
            pred_ranges = model(input_range_images, input_queries)
            loss = criterion(pred_ranges, output_ranges)

            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item() # Use .item() to avoid accumulating tensors

        lr_scheduler.step()
        
        # Logging (Rank 0 only inside function)
        print_log(epoch, loss_sum, loss_sum / len(train_loader), directory=model_directory)
        save_check_point(epoch, period=100)

def train_pixel_based_network():
    for epoch in range(epoch_start, epoch_end, 1):
        train_loader.sampler.set_epoch(epoch)
        
        loss_sum = 0.0
        iterator = tqdm(train_loader, leave=False, desc='train') if RANK == 0 else train_loader

        for low_res, high_res in iterator:
            low_res = low_res.cuda(non_blocking=True)
            high_res = high_res.cuda(non_blocking=True)

            optimizer.zero_grad()

            low_res = (low_res + 1.0) * 0.5 
            pred_high_res = model(low_res)
            pred_high_res = (pred_high_res * 2.0) - 1.0

            loss = criterion(pred_high_res, high_res)

            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().item()

        lr_scheduler.step()
        print_log(epoch, loss_sum, loss_sum / len(train_loader), directory=model_directory)
        save_check_point(epoch, period=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LiDAR super-resolution network")
    parser.add_argument('-c', '--config', type=str, required=True, help='Configuration filename. [.yaml]')
    parser.add_argument('-b', '--batch', type=int, required=False, default=16, help='Batch size per GPU.')
    parser.add_argument('-cp', '--checkpoint', type=str, required=False, default=None, help='Check point filename. [.pth]')
    args = parser.parse_args()

    # --- DDP Initialization ---
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        RANK = dist.get_rank()
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        # Fallback for single GPU/CPU debugging
        print("Warning: Not running via torchrun. Defaulting to single device.")
        local_rank = 0
        RANK = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------------------

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Dataset Config
    batch_size = args.batch
    if config['dataset']['name'].lower() == 'nuscenes':
        tmp_dir = os.environ.get("TMPDIR")
        use_work_dir = os.environ.get("USE_WORK", "0")
        if tmp_dir is not None and use_work_dir != "1":
            config['dataset']['args']['directory'] = os.path.join(tmp_dir, "nusc_dataset")
        config['dataset']['args']['directory'] += '/train_rv'
        config['dataset']['args']['nusc'] = True
        
        if RANK == 0:
            print("Nuscenes root directory:", config['dataset']['args']['directory'])

    train_dataset = generate_dataset(config['dataset'])

    # --- DDP Sampler & Loader ---
    # Sampler ensures each GPU gets a unique slice of data
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    
    # Num workers: 4-8 is usually good. pin_memory=True is crucial for speed.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Sampler handles shuffling
        drop_last=True, 
        num_workers=4, 
        pin_memory=True, 
        sampler=train_sampler
    )

    model = generate_model(config['model']['name'], config['model']['args']).cpu()
    epoch_start = 0

    # 2. Load Checkpoint BEFORE DDP (Solves the Key Mismatch)
    #    Since the model is not yet wrapped, it expects "encoder...", 
    #    and your checkpoint has "encoder...", so they match perfectly.
    if args.checkpoint is not None:
        if RANK == 0: 
            print(f"Loading checkpoint: {args.checkpoint}")
        
        # CRITICAL: map_location ensures weights load to the correct local GPU
        check_point = torch.load(args.checkpoint, map_location='cpu')
        
        if is_valid_check_point():
            # No need for complex key renaming logic anymore!
            model.load_state_dict(check_point['model']['state_dict'])
            
            epoch_start = check_point['epoch']
            # We delay loading optimizer/scheduler until they are initialized below
        else:
            if RANK == 0: print('ERROR: Invalid check point file:', args.checkpoint)
            exit(0)
    model.to(device)
    # 3. Wrap Model with DDP
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # 4. Initialize Optimizer & Scheduler
    optimizer = optim.Adam(params=list(model.parameters()), lr=1e-4)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.5)
    criterion = nn.L1Loss().to(device)

    # 5. Load Optimizer State (If Resuming)
    if args.checkpoint is not None and is_valid_check_point():
        optimizer.load_state_dict(check_point['optimizer'])
        lr_scheduler.load_state_dict(check_point['lr_scheduler'])
    epoch_end = 1000

    # Setup directories
    model_name = config['model']['name']
    model_directory = config['model']['output']
    
    if RANK == 0:
        os.makedirs(model_directory, exist_ok=True)
        print("=================== Training Configuration ====================  ")
        print('  Model:', model_name)
        print('  Output directory:', model_directory)
        print('  Batch (per GPU):', batch_size)
        print('  World Size:', dist.get_world_size() if dist.is_initialized() else 1)
        print("===============================================================  \n")

    if config['dataset']['type'] == 'range_images':
        train_pixel_based_network()
    elif config['dataset']['type'] == 'range_samples_from_image':
        train_implicit_network()
        
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()