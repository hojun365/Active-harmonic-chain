# 2023.12.20 Wed edited
import gc
import copy
import math
import time
import torch
torch.set_default_dtype(torch.float64)
import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
import torch.nn as nn
from torchsde import sdeint, BrownianInterval

class ODE(nn.Module):
    def __init__(self,nptl:int,mu:float,spring:float,tau:float):
        super(ODE,self).__init__()
        self.N = nptl   # the number of particle
        self.μ = mu     # motility
        self.τ = tau    # self propulsion decay rate
        self.K = spring # spring constant

    def forward(self,X):
        with torch.no_grad():
            rn = X[...,:self.N]; an = X[...,self.N:]
            rnp = torch.roll(rn,shifts=(0,-1),dims=(0,1)); rnm = torch.roll(rn,shifts=(0,+1),dims=(0,1))
            dr1 = rnp - rn; dr2 = rn - rnm
            Fr = self.μ*self.K(dr1-dr2)+an; Fa = -an/self.τ
        return torch.cat([Fr,Fa],dim=-1)

class SDE(nn.Module):
    sde_type = "stratonovich"; noise_type = "general"
    def __init__(self,ode,nptl:int,kbT:float,Da:float,diffusion_matrix):
        super(SDE,self).__init__()
        self.F = ode
        self.N = nptl             # the number of particle
        self.kbT = kbT            # thermal fluctuation
        self.Da = Da              # self-propulsion diffusion coefficient
        self.D = diffusion_matrix # diffusion matrix (it is matrix)

    def drift(self,t,X):
        return self.F(X)
    
    def diffusion(self,t,X):
        return self.D.expand([X.shape[0],2*self.N,2*self.N])

def Run(dt,T,X_initial,mu,spring,tau,kbT,Da,diffusion_matrix,device):
    # X_initial.shape = (n_traj, 2*nptl)
    # dt : time step, T : time length
    n_traj = len(X_initial[:,0]); n_ptl = int(len(X_initial[0,:])/2)
    # Time : initialization time
    Time = dt*torch.tensor([t for t in range(int(T/dt)+1)]).to(device)
    # what system I want to solve?
    ode = ODE(n_ptl,mu,spring,tau).to(device); sde = SDE(ode,n_ptl,kbT,Da,diffusion_matrix).to(device)
    if   sde.sde_type == "ito":          sde_method = "euler"
    elif sde.sde_type == "stratonovich": sde_method = "heun"
    # Solve the stochastic differential equations
    bm = BrownianInterval(t0=0.,t1=T+dt,dt=dt,size=(n_traj,2*n_ptl),device=device,cache_size=None)
    X_final = sdeint(sde,X_initial,Time,bm=bm,dt=dt,method=sde_method,names={'drift':'drift','diffusion':'diffusion'})[-1]
    return copy.deepcopy(X_final)