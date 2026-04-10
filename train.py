import os
import argparse
from tqdm import tqdm
from time import time

# Setup argparse to handle command-line arguments
parser = argparse.ArgumentParser(description='Train a deep learning model.')
parser.add_argument('--gpu', type=str, required=True, help='CUDA visible devices')
parser.add_argument('--dataset', type=str, required=True, choices=['Mine'], help='Dataset to use')
parser.add_argument('--crop_size', type=int, required=True, help='Crop size (width height)')
parser.add_argument('--model', type=str, required=True, 
                    choices=['LinkNet50', 'BAM_LinkNet50', 'ConvNeXt_LinkNet', 'LinkNet50_HK', 'BAM_LinkNet50_HK', 'MHNet', 'SwinT_MHNet', 'SwinT_LinkNet'], 
                    help='Model to use')
parser.add_argument('--exper_name', type=str, required=True, help='Name of the experiment')
parser.add_argument('--batch_size_card', type=int, required=True, help='Batch size per GPU card')
parser.add_argument('--seed', type=int, required=True, help='Random seed')
parser.add_argument('--epoch', type=int, required=True, help='Number of epochs')
parser.add_argument('--num_classes', type=int, default=1, help='classes')
parser.add_argument('--height_with', type=bool, default=False)
args = parser.parse_args()

# Set the environment variable for CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
from networks.dinknet import LinkNet50, BAM_LinkNet50, ConvNeXt_LinkNet, SwinT_LinkNet
# from networks_samroad.samroad import SAMRoad
from networks.MHNet import MHNet, LinkNet50_HK, BAM_LinkNet50_HK, SwinT_MHNet
from framework import MyFrame
from loss import dice_bce_loss, dice_ce_loss, dice_bce_mse_loss
from minedataset import Mine_Dataset

# Set seed for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Initialize model based on command-line argument
# Dictionary to map model names to classes
model_dict = {
    'LinkNet50': LinkNet50,
    'BAM_LinkNet50': BAM_LinkNet50,
    'ConvNeXt_LinkNet':ConvNeXt_LinkNet,
    'LinkNet50_HK': LinkNet50_HK,
    'BAM_LinkNet50_HK': BAM_LinkNet50_HK,
    'MHNet':MHNet,
    'SwinT_MHNet':SwinT_MHNet,
    'SwinT_LinkNet':SwinT_LinkNet
}

if args.num_classes==1:
    loss_select = dice_bce_loss
else:
    loss_select = dice_ce_loss
if args.model == 'LinkNet50_HK' or args.model == 'BAM_LinkNet50_HK' or args.model == 'MHNet' or args.model == 'SwinT_MHNet':
    loss_select = dice_bce_mse_loss

if args.model in model_dict:
    solver = MyFrame(model_dict[args.model], loss_select, 2e-4, args.num_classes)
else:
    raise ValueError(f"Unknown model: {args.model}")


batchsize = torch.cuda.device_count() * args.batch_size_card

# Initialize dataset based on command-line argument
if args.dataset == 'Mine':
    # train_dir = os.path.join(, 'train/')
    dataset = Mine_Dataset(args.crop_size, seed=args.seed, type='train', height_mask_out=args.height_with)
# elif args.dataset == 'LSRV2':
#     dataset = LSRV2_Dataset('/data1/yry22/Occlusion/dataset/',args.crop_size, seed=args.seed, type='train')
else:
    raise ValueError(f"Unknown dataset: {args.dataset}")
# Data loader setup
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=8
)

# Log file setup
os.makedirs('experiments/{}/'.format(args.exper_name),exist_ok=True)
os.makedirs('experiments/{}/weights/'.format(args.exper_name), exist_ok=True)

mylog = open('experiments/{}/'.format(args.exper_name) + 'train.log', 'w')
tic = time()
no_optim = 0
train_epoch_best_loss = 100.

# Training loop
for epoch in range(0, args.epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in tqdm(data_loader_iter):
        # import pdb
        # pdb.set_trace()
        # print(mask.min())
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
        # print('check problem:{}'.format(train_loss))
    train_epoch_loss /= len(data_loader_iter)
    
    # Log progress
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', args.crop_size, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', args.crop_size)

    # Save model and adjust learning rate based on conditions
    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('experiments/{}/weights/'.format(args.exper_name) + args.exper_name + '.th')
        
    if epoch in [int(0.5*args.epoch), int(0.7*args.epoch), int(0.85*args.epoch)]:
        solver.update_lr(5.0, factor=True, mylog=mylog)
    
    if epoch == args.epoch:
        solver.save('experiments/{}/weights/'.format(args.exper_name) + args.exper_name + '_{}.th'.format(args.epoch))
    
    mylog.flush()

# Finalize logging
print('Finish!', file=mylog)
print('Finish!')
mylog.close()
