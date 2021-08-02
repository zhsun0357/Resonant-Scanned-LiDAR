import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy
from torch.autograd import Variable
import copy as cp
import pdb

dtype = torch.float32

## define torch layer to generate scanning patterns.
"""
input: args, alpha, beta, gamma, delta, 
        batch_data: 'mdi', 'd'
output: batch_data: 'mdi', 'd'(sparsified), 'gt'(originally 'd')
"""

class Pattern_Generator_grid(nn.Module):
    def __init__(self, samples, frequencies):
        super(Pattern_Generator_grid, self).__init__()
        
        self.N = samples
        self.m = frequencies

        ## parameters initialization
        ## gaussian random init
        self.alpha = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.beta = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.gamma = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.delta = Variable((torch.rand(self.m) - 0.5).type(dtype))

        """
        alpha0 = torch.from_numpy(np.load('perturb_alpha.npy')).type(dtype)
        gamma0 = torch.from_numpy(np.load('perturb_gamma.npy')).type(dtype)
        beta0 = torch.from_numpy(np.load('perturb_beta.npy')).type(dtype)
        delta0 = torch.from_numpy(np.load('perturb_delta.npy')).type(dtype)

        self.alpha = Variable((torch.zeros(self.m)).type(dtype))
        self.beta = Variable((torch.zeros(self.m)).type(dtype))
        self.gamma = Variable((torch.zeros(self.m)).type(dtype))
        self.delta = Variable((torch.zeros(self.m)).type(dtype))

        self.alpha[np.arange(5)*10] = alpha0
        self.gamma[np.arange(5)*10] = gamma0
        self.beta[np.arange(5)*10] = beta0
        self.delta[np.arange(5)*10] = delta0
        """

        """ TODO: needs to change!!
        """
        self.base_freqx = 120
        self.base_freqy = 240
        self.delta_freqx = 10
        self.delta_freqy = 20
        self.frx = 1.0
        self.fry = 2.0
        self.Q = 20

        self.fx = (np.arange(self.m)*self.delta_freqx + self.base_freqx)/ 140
        self.fy = (np.arange(self.m)*self.delta_freqy + self.base_freqy)/ 140
        self.Hx, self.Hy = self.transfer_func(self.fx, self.fy)
        
        # pdb.set_trace()
        self.Cx = np.zeros([self.m, self.N, self.m])
        self.Cy = np.zeros([self.m, self.N, self.m])
        self.Sx = np.zeros([self.m, self.N, self.m])
        self.Sy = np.zeros([self.m, self.N, self.m])

        for mm in np.arange(self.m):
            self.t = np.arange(self.N).astype(np.float32)/(self.N/7)*2*np.pi + mm*14*np.pi
            self.Cx[mm,:,:] = np.cos(np.outer(self.t, self.fx))
            self.Cy[mm,:,:] = np.cos(np.outer(self.t, self.fy))
            self.Sx[mm,:,:] = np.sin(np.outer(self.t, self.fx))
            self.Sy[mm,:,:] = np.sin(np.outer(self.t, self.fy))

        # pdb.set_trace()
        
        self.fixed_params = ['fx', 'fy', 'Hx', 'Hy', 'Cx', 'Cy', 'Sx', 'Sy']
        self.train_params = ['alpha', 'gamma', 'beta', 'delta']

        for param in self.fixed_params:
            self.__dict__[param] = torch.from_numpy(self.__dict__[param]).type(dtype)
            self.__dict__[param].requires_grad_(requires_grad = False)
        for param in self.train_params:
            self.__dict__[param] = self.__dict__[param].type(dtype)
            self.__dict__[param].requires_grad_(requires_grad = True)

        self.grad_list = []
        ## shape: (m, N, m)

    def transfer_func(self, fx, fy):
        Hx = 1/np.sqrt(((fx/self.frx)**2 - 1)**2 + (fx/self.frx/self.Q)**2) / self.Q
        Hy = 1/np.sqrt(((fy/self.fry)**2 - 1)**2 + (fy/self.fry/self.Q)**2) / self.Q
        return Hx, Hy

    def print_grad(self, grad):
        """
        self.grad_list.append(grad)
        if torch.sum(grad) == 0:
            print("error!")
            pdb.set_trace()
        else:
            pass
        """
        pass

    def forward(self):
        ## alpha, beta, gamma, delta: (1,1,m)  
        
        """
        alpha = params['alpha'].type(dtype)
        gamma = params['gamma'].type(dtype)
        beta = params['betaa'].type(dtype)
        delta = params['delta'].type(dtype)
        """

        """
        Since we want the whole energy consumption to be bounded, we
        should have alpha**2 + gamma**2 = 1, beta**2 + delta**2 = 1
        Also add transfer function
        """
        Ax_norm = torch.sqrt(torch.norm(self.alpha)**2 + torch.norm(self.gamma)**2)
        Ay_norm = torch.sqrt(torch.norm(self.beta)**2 + torch.norm(self.delta)**2)

        if Ax_norm >= 1:
            alpha_norm = (self.alpha* self.Hx).view(1,self.m,1).repeat(self.m,1,1)/Ax_norm 
            gamma_norm = (self.gamma* self.Hx).view(1,self.m,1).repeat(self.m,1,1)/Ax_norm
        else:
            alpha_norm = (self.alpha* self.Hx).view(1,self.m,1).repeat(self.m,1,1)
            gamma_norm = (self.gamma* self.Hx).view(1,self.m,1).repeat(self.m,1,1)

        if Ay_norm >= 1:
            beta_norm = (self.beta* self.Hy).view(1,self.m,1).repeat(self.m,1,1)/Ay_norm
            delta_norm = (self.delta* self.Hy).view(1,self.m,1).repeat(self.m,1,1)/Ay_norm
        else:
            beta_norm = (self.beta* self.Hy).view(1,self.m,1).repeat(self.m,1,1)
            delta_norm = (self.delta* self.Hy).view(1,self.m,1).repeat(self.m,1,1)
        
        """
        alpha_norm.register_hook(self.print_grad)
        beta_norm.register_hook(self.print_grad)
        gamma_norm.register_hook(self.print_grad)
        delta_norm.register_hook(self.print_grad)
        """

        # target = target.repeat(self.m, 1, 1, 1)
        # pdb.set_trace()
        grid_x = torch.matmul(self.Cx, alpha_norm) + torch.matmul(self.Sx, gamma_norm)
        grid_y = torch.matmul(self.Cy, beta_norm) + torch.matmul(self.Sy, delta_norm)
        """ TODO: normalize if max(abs) larger than 1
        """
        if torch.max(torch.abs(grid_x)) >= 1.0/1.02:
            grid_x = (grid_x - torch.mean(grid_x)) / (torch.max(torch.abs(grid_x)) - torch.mean(grid_x)) / 1.02
        if torch.max(torch.abs(grid_y)) >= 1.0/1.02:
            grid_y = (grid_y - torch.mean(grid_y)) / (torch.max(torch.abs(grid_y)) - torch.mean(grid_y)) / 1.02
        ## grid_x, grid_y: (m,N,1), range: (-1,1)
        
        # pdb.set_trace()
        ## generate point cloud
        # grid = torch.cat((grid_y, grid_x), 2).unsqueeze(3).permute(0,1,3,2).cuda()
        # pdb.set_trace()
        # pdb.set_trace()
        # xout = F.grid_sample(target, grid, mode = 'bilinear').squeeze()
        ## xout: (m,N), grid: (m, N, 1,2), target: (m, 1, H0, W0)

        return grid_x.squeeze(), grid_y.squeeze()


