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

class Pattern_Generator_grid(nn.Module):
    def __init__(self, samples, frequencies, check_grad = False):
        super(Pattern_Generator_grid, self).__init__()
        
        self.N = samples
        self.m = frequencies
        self.check_grad = check_grad

        ## parameters initialization
        ## gaussian random init
        self.alpha = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.beta = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.gamma = Variable((torch.rand(self.m) - 0.5).type(dtype))
        self.delta = Variable((torch.rand(self.m) - 0.5).type(dtype))
        
        self.base_freqx = 120
        self.base_freqy = 120
        self.delta_freqx = 10
        self.delta_freqy = 10
        self.frx = 1.0
        self.fry = 1.0
        self.Q = 20

        self.fx = (np.arange(self.m)*self.delta_freqx + self.base_freqx)/ 140
        self.fy = (np.arange(self.m)*self.delta_freqy + self.base_freqy)/ 140
        self.Hx, self.Hy = self.transfer_func(self.fx, self.fy)
        
        self.Cx = np.zeros([self.m, self.N, self.m])
        self.Cy = np.zeros([self.m, self.N, self.m])
        self.Sx = np.zeros([self.m, self.N, self.m])
        self.Sy = np.zeros([self.m, self.N, self.m])

        for mm in np.arange(self.m):
            self.t = np.arange(self.N).astype(np.float32)/(self.N)*14*np.pi + mm*14*np.pi
            self.Cx[mm,:,:] = np.cos(np.outer(self.t, self.fx))
            self.Cy[mm,:,:] = np.cos(np.outer(self.t, self.fy))
            self.Sx[mm,:,:] = np.sin(np.outer(self.t, self.fx))
            self.Sy[mm,:,:] = np.sin(np.outer(self.t, self.fy))
        
        self.fixed_params = ['fx', 'fy', 'Hx', 'Hy', 'Cx', 'Cy', 'Sx', 'Sy']
        self.train_params = ['alpha', 'gamma', 'beta', 'delta']

        for param in self.fixed_params:
            self.__dict__[param] = torch.from_numpy(self.__dict__[param]).type(dtype)
            self.__dict__[param].requires_grad_(requires_grad = False)
        for param in self.train_params:
            self.__dict__[param] = self.__dict__[param].type(dtype)
            self.__dict__[param].requires_grad_(requires_grad = True)

        self.grad_list = {}
        ## shape: (m, N, m)

    def transfer_func(self, fx, fy):
        Hx = 1/np.sqrt(((fx/self.frx)**2 - 1)**2 + (fx/self.frx/self.Q)**2) / self.Q
        Hy = 1/np.sqrt(((fy/self.fry)**2 - 1)**2 + (fy/self.fry/self.Q)**2) / self.Q
        return Hx, Hy

    def save_grad(self, name, grad):
        self.grad_list[name] = grad

    def forward(self): 
        """
        Since we want the RMS actuation to be bounded, we
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
        
        grid_x = torch.matmul(self.Cx, alpha_norm) + torch.matmul(self.Sx, gamma_norm)
        grid_y = torch.matmul(self.Cy, beta_norm) + torch.matmul(self.Sy, delta_norm)

        if torch.max(torch.abs(grid_x)) >= 1.0/1.02:
            grid_x = (grid_x - torch.mean(grid_x)) / (torch.max(torch.abs(grid_x)) - torch.mean(grid_x)) / 1.02
        if torch.max(torch.abs(grid_y)) >= 1.0/1.02:
            grid_y = (grid_y - torch.mean(grid_y)) / (torch.max(torch.abs(grid_y)) - torch.mean(grid_y)) / 1.02
        
        if self.check_grad:
            grid_x.register_hook(lambda grad, name="grid_x": self.save_grad(name, grad))
            grid_y.register_hook(lambda grad, name="grid_y": self.save_grad(name, grad))

        return grid_x.squeeze(), grid_y.squeeze()


