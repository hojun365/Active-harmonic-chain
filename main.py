# 2024.11.01 Fri
## position을 pbc를 반영하여 sine/cosine encoding을 해보자
import os
import copy
import math
import torch
import pickle
import argparse
import warnings
import Model as MM
import numpy as np
import torch.nn as nn
import AnalyticValue as AV
import torch.optim as optim
from tqdm import trange
from torchsde import BrownianInterval
warnings.filterwarnings(action="ignore")
#-----------------------------------------------------------------------------------------------#
# Parser Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--N",type=int,default=3)
parser.add_argument("--device",type=str,default="cpu")
parser.add_argument("--epoch",type=int,default=3000)
parser.add_argument("--hid",type=int,default=20) # fully-connected보다 0.9배의 parameter를 사용해야 comparable
parser.add_argument("--block",type=int,default=1) 
parser.add_argument("--lmbda",type=float)
parser.add_argument("--dt",type=float,default=0.005)
parser.add_argument("--tstep",type=int,default=1000)
parser.add_argument("--m",type=str,default="e") # "e" for equivariant model/ "c" for fully-connected model
args = parser.parse_args()
dt = args.dt; tstep = args.tstep
device = args.device; n_hid = args.hid; lmbda = args.lmbda; N = args.N; epochs = args.epoch
n_blocks = args.block; types = args.m
PATH = os.getcwd()
if not os.path.exists(PATH): os.mkdir(PATH)
if not os.path.exists(PATH+"/data/"): os.mkdir(PATH+"/data/")
if not os.path.exists(PATH+"/midtrain/"): os.mkdir(PATH+"/midtrain/")
#------------------------------------------------------------------------------------------------------------------------------------#
def F(X):
    r = X[...,:N]; a = X[...,N:]
    dr = torch.roll(r,shifts=(0,-1),dims=(0,-1))-r+PBC
    Fr = μ*K*(dr - torch.roll(dr,shifts=(0,+1),dims=(0,-1)))+a; Fa = -a/τ
    return torch.cat([Fr, Fa],dim=-1)

def Invar(X):
    X = X - X.mean(dim=-1,keepdim=True).detach()
    return X

def Pencode(X):
    X = Invar(X)
    return torch.cat([X//L,X%L],-1) # domain, pbc 上 position, shape = (...,2N)

def Noise(dW, diffusion_matrix):
    return torch.matmul(dW,diffusion_matrix)
#------------------------------------------------------------------------------------------------------------------------------------#
# Parameters
ensemble = 100; lr = .01; T = dt*tstep; Density = 1.0 # density는 1로 유지하자 (입자간 평균 거리가 1이 되도록)
L = 1.0; μ = 1.0; τ = 1.0; K = 10.0; Da = 10.0; kbT = 0.2
# diffusion matrix
diffusion_matrix = torch.zeros(size=(2*N,2*N)).to(device)
diffusion_matrix[:N,:N] = math.sqrt(2*μ*kbT)*torch.eye(N).to(device); diffusion_matrix[N:,N:] = math.sqrt(2*Da/(τ**2))*torch.eye(N).to(device)
#------------------------------------------------------------------------------------------------------------------------------------#
# Initial condition
r_i = L*torch.tensor([n/N for n in range(N)]).expand([ensemble,N])
a_i = torch.normal(mean=0,std=math.sqrt(2*Da/(τ**2)), size=(ensemble,N))
X_i = torch.cat([r_i,a_i],-1).to(device)
# Periodic boundary condition
PBC = torch.zeros(size=(ensemble,N)).to(device)
PBC[:,-1:] = L*torch.ones(size=(ensemble,1)).to(device)
#------------------------------------------------------------------------------------------------------------------------------------#
# Neural network
if   types == "c": NN = MM.FCN(N,n_hid,n_blocks).to(device); name = "Convention"
elif types == "e": NN = MM.Equiv(N,int(n_hid*0.9),n_blocks).to(device); name = "Equivariant"
Optimizer = optim.AdamW(NN.parameters(),lr=lr); Optimizer.zero_grad()
torch.save(copy.deepcopy(NN).cpu().state_dict(), PATH+"/midtrain/"+name+"_Neural_{0}ptc_{1}hid_{2}block_{3}lmbda_{4}epoch.pth".format(N,n_hid,n_blocks,lmbda,0))
#------------------------------------------------------------------------------------------------------------------------------------#
# train neural network
ψ_T = []; G_T = []; W_T = []
AWork = AV.AW(N,τ,μ,K,Da,kbT,lmbda); ALoss = -AV.CGF(N,τ,μ,K,Da,kbT,lmbda); AGirs = lmbda*AWork + ALoss
for epoch in trange(epochs,desc="TRAIN λ = {0:.2f}. Target Loss : {1:.2f}, Girs : {2:.2f}, Work : {3:.2f}".format(lmbda,ALoss,AGirs,AWork)):
    # Generate trajectory
    Work = 0; Girs = 0
    bm = BrownianInterval(t0=0.,t1=dt*tstep,dt=dt,size=(ensemble,2*N),cache_size=None,device=device)
    for t in range(tstep):
        # Force
        Fo = F(X_i); δu = NN(torch.cat([Pencode(X_i[...,:N].detach()),X_i[...,N:].detach()],-1))
        # Euelr-Murayama
        X_t = X_i + dt*(Fo+δu) + Noise(bm(dt*t,dt*(t+1)),diffusion_matrix)
        # Caclulate loss
        dr = (X_t-X_i)[...,:N]; am = 1/2*(X_t+X_i)[...,N:]
        Work += 1/μ*(am*dr).sum(-1)/T
        Girs += dt/2*(δu[...,:N]**2/(2*μ*kbT)+δu[...,N:]**2/(2*Da/(τ**2))).sum(-1)/T
        # update the condition
        X_i = X_t

    # Loss
    Loss = Girs - lmbda*Work
    Loss.mean().backward()
    # nn.utils.clip_grad_norm_(NN.parameters(),max_norm=1.0)
    Optimizer.step()
    Optimizer.zero_grad()
    X_i = X_i.detach()

    # store data
    ψ_T.append(Loss.clone().detach().cpu().mean().item())
    G_T.append(Girs.clone().detach().cpu().mean().item())
    W_T.append(Work.clone().detach().cpu().mean().item())
    print("Loss : {0:.3f} = {1:.3f} - {3} * {2:.3f}".format(ψ_T[-1],G_T[-1],W_T[-1],lmbda))

    # midtrain 모델 저장
    if (epoch+1)%(epochs//100) == 0:
        torch.save(copy.deepcopy(NN).cpu().state_dict(), PATH+"/midtrain/"+name+"_Neural_{0}ptc_{1}hid_{2}block_{3}lmbda_{4}epoch.pth".format(N,n_hid,n_blocks,lmbda,epoch+1))    
    
    if   np.isnan(ψ_T[-1]) == True:  raise Exception("Diverge")
    elif np.isnan(ψ_T[-1]) == False: continue

# save result
data_path = PATH+"/data/"
with open(data_path+name+"_Loss_{0}ptc_{1}hid_{2}block_{3}lmbda_{4}epoch.pkl".format(N,n_hid,n_blocks,lmbda,epochs),"wb") as fa: pickle.dump(ψ_T,fa)
with open(data_path+name+"_Obse_{0}ptc_{1}hid_{2}block_{3}lmbda_{4}epoch.pkl".format(N,n_hid,n_blocks,lmbda,epochs),"wb") as fb: pickle.dump(W_T,fb)
with open(data_path+name+"_Girs_{0}ptc_{1}hid_{2}block_{3}lmbda_{4}epoch.pkl".format(N,n_hid,n_blocks,lmbda,epochs),"wb") as fc: pickle.dump(G_T,fc)
