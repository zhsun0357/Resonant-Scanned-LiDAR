import os
import argparse
import torch
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F

from pattern_generator import Pattern_Generator_grid
from fast_bilateral_solver import BilateralSolver

import numpy as np
from tqdm import tqdm
from operator import itemgetter
import json
import matplotlib.pyplot as plt
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

dtype = torch.float32

parser = argparse.ArgumentParser(description='Monocular Depth Refinement')
parser.add_argument('--save_directory',
                    type=str,
                    default='train_logging',
                    help='directory to save results in')
parser.add_argument('--data_dir',
                    type=str,
                    default='demo_data',
                    help='directory to load data')
parser.add_argument('--num_iters',
                    default=2000,
                    type=int,
                    help='number of optimization iterations')
parser.add_argument('--lr',
                    default=1e-3,
                    type=float,
                    help='learning rate')
parser.add_argument('--samples',
                    default=200,
                    type=int,
                    help='number of sparse samples')
parser.add_argument('--frequencies',
                    default=5,
                    type=int,
                    help='number of frequency components')
parser.add_argument('--num_frames',
                    default=5,
                    type=int,
                    help='number of frames to optimize for')
parser.add_argument('--save_every',
                    default=100,
                    type=int,
                    help='steps to save an intermediate result')
parser.add_argument('--device',
                    default="cuda",
                    type=str,
                    help='device to run on')


args = parser.parse_args()

if not os.path.exists(args.save_directory):
    os.mkdir(args.save_directory)
    
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
args.save_directory = os.path.join(args.save_directory,
                    "samples={}.frames={}.freqs={}.lr={}.time={}".
                    format(args.samples, args.num_frames, args.frequencies,
                    args.lr, current_time))
if not os.path.exists(args.save_directory):
    os.mkdir(args.save_directory)

writer = SummaryWriter(os.path.join(args.save_directory, "logfile"))

print(args)
device = torch.device(args.device)

def rec_error(batch_data):
    loss = 0
    M = 10 ## resolution of RoI weight map
    rel_error = torch.abs(batch_data['bsol'] - batch_data['gt']) / batch_data['gt'] * 10
    """
    use relative error as RoI weight, this is not practical in real-world applications since
    ground truth depth (and therefore relative error) is unknown. We just use this setting as 
    to demonstrate the convergence process of the optimization framework.
    User can specify any float value RoI weight as they want.
    """
    _, _, H, W = rel_error.shape
    for ii in np.arange(M):
        for jj in np.arange(M):
            center_x = (ii - M/2)/(M/2)
            center_y = (jj - M/2)/(M/2)

            low_x = int(ii/(M+1) * H)
            high_x = int((ii + 1)/(M+1) * H)
            low_y = int(jj/(M+1) * W)
            high_y = int((jj+ 1)/(M+1) * W)
            weight = torch.mean(torch.mean(rel_error[:, 0, low_x:high_x, low_y:high_y], dim = 1), dim = 1)

            closest_dis, closest_idx = torch.min(torch.sqrt((batch_data['grid_x'] - center_x)**2 + (batch_data['grid_y'] - center_y)**2), dim = 1)
            out_patch = (1 - (closest_dis.detach() <= 1/M)).type(dtype)
            loss += torch.sum(weight * out_patch * (batch_data['grid_x'][np.arange(args.frequencies), closest_idx.cpu().numpy()] - center_x)**2) + \
                    torch.sum(weight * out_patch * (batch_data['grid_y'][np.arange(args.frequencies), closest_idx.cpu().numpy()] - center_y)**2)
    return loss/args.frequencies/(M*M)


def load_data(args):
    """
    load a single image for RoI optimization
    user can also load multiple images
    """
    batch_data = {}
    batch_data['gt'] = torch.from_numpy(np.load(os.path.join(args.data_dir, 'gt.npy'))).float()
    batch_data['mdi'] = torch.from_numpy(np.load(os.path.join(args.data_dir, 'mdi.npy'))).float()
    batch_data['gt'] = batch_data['gt'].unsqueeze(0).unsqueeze(1).repeat(args.num_frames, 1, 1, 1).cuda().detach()
    batch_data['mdi'] = batch_data['mdi'].unsqueeze(0).unsqueeze(1).repeat(args.num_frames, 1, 1, 1).cuda().detach()
    return batch_data

def run_optimize(args, pg, bsolver, optimizer):
    pg.train()
    batch_data = load_data(args)
    
    for i in tqdm(range(args.num_iters)):
        batch_data['grid_x'], batch_data['grid_y'] = pg()
        batch_data['grid_x'].requires_grad_(requires_grad = True)
        batch_data['grid_y'].requires_grad_(requires_grad = True)

        _, _, H, W = batch_data['gt'].shape
        points = torch.cat((batch_data['grid_x'].unsqueeze(2), batch_data['grid_y'].unsqueeze(2)), dim = 2).detach()
        points = (points + 1)/2
        """
        To optimize the pattern for all frames, num_frames should be
        longer than the physical pattern repeating period.
        points: (num_frames, samples, 2(x,y)), all value between 0 to 1
        depth_gt: (num_frames, 1, H, W)
        monocular_depth: (num_frames, 1, H, W)
        """
        batch_data['bsol'], _ = bsolver.solve(points, batch_data['gt'], batch_data['mdi'])
        for key, val in batch_data.items():
            batch_data[key] = val.to(device)
        
        loss = rec_error(batch_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('data/train_loss', loss.item(), i)

        if(i == 0):
            gt_save = np.save(os.path.join(args.save_directory, 'figure_gt.npy'), batch_data['gt'].squeeze().detach().cpu().numpy())
        if (i % args.save_every == 0):
            np.save(os.path.join(args.save_directory, 'grid_x_grad_{}.npy'.format(i)), pg.grad_list['grid_x'].squeeze().detach().cpu().numpy())
            np.save(os.path.join(args.save_directory, 'grid_y_grad_{}.npy'.format(i)), pg.grad_list['grid_y'].squeeze().detach().cpu().numpy())
            np.save(os.path.join(args.save_directory, 'grid_x_{}.npy'.format(i)), batch_data['grid_x'].squeeze().detach().cpu().numpy())
            np.save(os.path.join(args.save_directory, 'grid_y_{}.npy'.format(i)), batch_data['grid_y'].squeeze().detach().cpu().numpy())

def main():
    pg = Pattern_Generator_grid(args.samples, args.frequencies, check_grad = True)
    pg.to(device)

    optimizer = torch.optim.Adam(itemgetter(*pg.train_params)(pg.__dict__), lr=args.lr)
    bsolver = BilateralSolver()
    run_optimize(args, pg, bsolver, optimizer)

if __name__ == '__main__':
    main()
