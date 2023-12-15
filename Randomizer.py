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

    def crossing(self,r):
        """
        Checking particle crossing
        """
        with torch.no_grad():


    def forward(self,X): 
        """
        The total legnth of the system set as L = 1.
        """
        rn = X[...,:self.N]
        # r_{n+1},                                       # r_{n-1}
        rn_p1 = torch.roll(rn,shifts=(0,-1),dims=(0,1)); rn_m1 = torch.roll(rn,shifts=(0,+1),dims=(0,1)) 
        # r_{n+1} - r_{n},  # r_{n} - r_{n-1} 
        dr1 = (rn_p1-rn)%1; dr2 = (rn-rn_m1)%1 # 그냥 입자 사이의 거리
        # 부호를 그냥 날리면 큰일남(왜냐하면 crossing이 일어났을 때, 미는 역할을 하는 spring이 당기는 역할로 바뀜)
        # 부호를 찾아내자 (미는 역할일지 당기는 역할일지), 즉 crossing이 일어났는지 판단을 하자

        return 0

class SDE(nn.Module):
    def __init__(self,ode,kbT:float,Da:float):
        super(SDE,self).__init__()
        self.deterministic = ode
        self.kbT = kbT
        self.Da = Da
    
    def forward(self,X):
        # X.shape = (n_traj, 2N)
        return 0

def Run(시간, 초기조건, 파라미터 조건, cpu or GPU?):
    ode = ODE(); sde = SDE

    return 0