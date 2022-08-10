import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsde import BrownianInterval, sdeint
from tqdm import tqdm

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA: device="cuda"
else: device="cpu"
    
#Ritz block
class RitzBlock(nn.Module):
    def __init__(self, n_hid):
        super(RitzBlock, self).__init__()
        self.n_hid=n_hid
        self.activation=nn.Tanh().to(device)

        #Layer1
        self.Layer1_Weight=nn.Parameter(torch.normal(mean=0,std=1,size=(self.n_hid,self.n_hid),requires_grad=True).to(device))
        self.Layer1_Bias=nn.Parameter(torch.normal(mean=0,std=1,size=(1,self.n_hid),requires_grad=True).to(device))
        #Layer2
        self.Layer2_Weight=nn.Parameter(torch.normal(mean=0,std=1,size=(self.n_hid,self.n_hid),requires_grad=True).to(device))
        self.Layer2_Bias=nn.Parameter(torch.normal(mean=0,std=1,size=(1,self.n_hid),requires_grad=True).to(device))

    def layer1(self,X):
        return self.activation(F.linear(X,self.Layer1_Weight,self.Layer1_Bias)).to(device)
    def layer2(self,X):
        return self.activation(F.linear(X,self.Layer2_Weight,self.Layer2_Bias)).to(device)

    def forward(self,X):
        h1=self.layer1(X).to(device)
        h2=self.layer2(h1).to(device)
        return h2+X

#neural network
class NeuralNetworks(nn.Module):
    def __init__(self, n_block, n_hid):
        super(NeuralNetworks,self).__init__()
        self.n_block = n_block
        self.n_hid = n_hid
        self.dim_X = 4
        self.dim_input = self.dim_X+1

        self.pre_Weight = nn.Parameter(torch.normal(mean=0,std=1,size=(self.n_hid,self.dim_input), requires_grad=True).to(device))
        self.pre_Bias = nn.Parameter(torch.normal(mean=0,std=1,size=(1,self.n_hid), requires_grad=True).to(device))
        self.n_Weight = nn.Parameter(torch.normal(mean=0,std=0.1,size=(self.dim_X,self.n_hid), requires_grad=True).to(device))
        self.n_Bias = nn.Parameter(torch.normal(mean=0,std=0.1,size=(1,self.dim_X), requires_grad=True).to(device))

        self.Ritzblocks = nn.ModuleList().to(device)
        for _ in range(self.n_block): self.Ritzblocks.append(RitzBlock(n_hid).to(device))

    def forward(self, X, lambdas):
        XX=torch.cat([X,lambdas*torch.ones([*X.shape[:-1], 1]).to(device)],-1).to(device)
        if self.n_hid>self.dim_input:
            pad = nn.ConstantPad1d((0,n_hid-self.dim_input),0).to(device)
            o = pad(XX).to(device)
        else:
            o = F.linear(XX,self.p_Weight,self.p_Bias).to(device)
        for block in self.Ritzblocks:
            oo = block(o)
            o = oo.to(device)
        O=F.linear(o,self.n_Weight,self.n_Bias)
        return O.to(device)

class ODE(nn.Module):
    def __init__(self, spring_constant, Friction):
        super().__init__()
        self.k=spring_constant
        self.gamma=Friction[0]
        self.gammaR=Friction[-1]
    
    def forward(self, x):
        r=x[..., :2]; v=x[..., 2:]
        fr=v-self.k/self.gamma*r; fv=-self.gammaR*v
        return torch.cat([fr,fv], -1).to(device)

class SDE(nn.Module):
    sde_type='ito'
    noise_type='general'
    
    def __init__(self, drift0, diffusion,device=device):
        super(SDE, self).__init__()
        self.dim_X=4
        self._drift=drift0.to(device)
        self.DMatrix=diffusion.to(device)

    def drift(self, t, x):
        return NN(x,lambdas).to(device)+self._drift(x).to(device)
    
    def diffusion(self, t, x):
        return self.DMatrix.expand(x.size(0), self.dim_X, self.dim_X).to(device)
    
    def drift_0(self, t, x):
        return self._drift(x).to(device)

n_hid=200; n_block=3

kbT=0.05 #temperature
d=2; sigma=1; Fa=20
gamma=10; DT=kbT/gamma; DR=3*DT/sigma
gammaR=(d-1)*DR
DRp=(d-1)/d*DR*(Fa/gamma)**2

spring_constant=0
Friction=[gamma, gammaR]
diffusion=torch.zeros(size=(4,4)).to(device)
diffusion[:2,:2]=np.sqrt(2*DT)*torch.eye(2).to(device); diffusion[2:,2:]=np.sqrt(2*DRp)*torch.eye(2).to(device)

l=-10
Lam=torch.tensor([l])
dt=1e-3; tlength=5000; lr=5e-2
ntraj=1000
ntraj_total=ntraj*len(Lam)

Time=torch.arange(0, dt*(tlength+1)).to(device)
x_initial=torch.zeros(size=(ntraj_total,4)).to(device)
lambdas=Lam.repeat(ntraj,1).t().reshape([-1,1]).to(device)

ode=ODE(spring_constant, Friction).to(device)
sde=SDE(ode, diffusion).to(device)
sde_method = 'euler' if sde.sde_type == 'ito' else 'midpoint'

NN=NeuralNetworks(n_block, n_hid).to(device)
Optimizer=optim.SGD(NN.parameters(), lr)

losses=[]; work=[]; weight=[]
for _ in tqdm(range(30000), desc="Train"):
    Optimizer.zero_grad()
    with torch.no_grad():
        bm=BrownianInterval(t0=Time[0],t1=Time[-1],dt=dt, size=(ntraj_total,4),device=device)
        traj=sdeint(sde,x_initial,Time,dt=dt,bm=bm,method=sde_method,names={"drift":"drift", "diffusion":"diffusion"})

    delta_u=NN(traj,lambdas)
    K=1/(2*DT)*torch.sum(delta_u[:-1,:,:2]**2,(0,-1))*dt/(dt*tlength) + 1/(2*DRp)*torch.sum(delta_u[:-1,:,2:]**2,(0,-1))*dt/(dt*tlength)
    
    dr=traj[1:,:,:2]-traj[:-1,:,:2]; vm=1/2*(traj[:-1,:,2:]+traj[1:,:,2:])
    W=torch.sum(dr*vm, (0,-1))/(tlength*dt)
    
    loss_batch=(K - lambdas.squeeze()*W)
    loss = loss_batch.mean()

    assert not torch.isnan(loss), "We've got a NaN"

    loss.backward()

    x_init = traj[-1,...].detach()
    Optimizer.step()  

    losses.append(loss.detach().item())
    work.append(torch.mean(W).detach().item())
    weight.append(torch.mean(K).detach().item())
    
with open("{0}_loss.pkl".format(l), "wb") as fa: pickle.dump(losses, fa)
with open("{0}_work.pkl".format(l), "wb") as fb: pickle.dump(work, fb)
with open("{0}_weight.pkl".format(l), "wb") as fc: pickle.dump(weight, fc)
