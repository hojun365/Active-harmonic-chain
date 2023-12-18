# 2023.12.15 Fri edited
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
        self.μ = mu     # drift
        self.τ = tau    # self propulsion decay rate
        self.K = spring # spring constant

    def True_position(self,X):
        """
        Periodic boundary condition 고려 후의 위치 (거리 계산에 필요함)
        """
        with torch.no_grad():
            rn_PBC = X[...,:self.N]%system_size
        return rn_PBC

    def Domain(self,X):
        """
        (0, system_size): 0th domain, (-system_size,0): -1st domain, ..., (system_size, 2*system_suze): 1st domain
        """
        with torch.no_grad():
            rn = X[...,:self.N]
            domain_id = (rn/system_size).to(torch.int64) # 입자가 어떤 domain에 들어가 있는가?
        return domain_id

    def True_Distance(self,X):
        

            



        return rn

    

    def forward(self,X):
        with torch.no_grad():
            rn = X[...,:self.N]


        return 0

class SDE(nn.Module):
    def __init__(self,ode,kbT:float,Da:float):
        super(SDE,self).__init__()
        self.deterministic = ode
        self.kbT = kbT
        self.Da = Da
    
    def forward(self,X):
        return 0

def Run(시간, 초기조건, 파라미터 조건, cpu or GPU?):
    ode = ODE(); sde = SDE()
    Noise = BrownianInterval()
    final = sdeint(sde,)
    return copy.deepcopy(final)