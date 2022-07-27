import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings(action="ignore")
import numpy as np
from tqdm import tqdm
from torchsde import BrownianInterval, sdeint

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA: device="cuda:3"
else: device="cpu"

#Ritz Blocks
class RitzBlock(nn.Module):
    def __init__(self, n_hid):
        super(RitzBlock, self).__init__()
        self.n_hid = n_hid

        ##Layer 1
        self.Weight_layer1 = nn.Parameter(torch.zeros(size=(self.n_hid,self.n_hid),requires_grad=True).to(device))
        self.Bias_layer1 = nn.Parameter(torch.zeros(size=(1,self.n_hid),requires_grad=True).to(device))    
        ##Layer 2
        self.Weight_layer2 = nn.Parameter(torch.zeros(size=(n_hid,n_hid),requires_grad=True).to(device))
        self.Bias_layer2 = nn.Parameter(torch.zeros(size=(1,n_hid),requires_grad=True).to(device))

    def layer1(self,X): 
        return torch.tanh(torch.matmul(X, self.Weight_layer1)+self.Bias_layer1).to(device)
    def layer2(self,X): 
        return torch.tanh(torch.matmul(X, self.Weight_layer2)+self.Bias_layer2).to(device)
    
    def forward(self,X):
        h1 = self.layer1(X).to(device)
        h2 = self.layer2(h1).to(device)
        return h2 + X

#pre model
class Premodel(nn.Module):
    def __init__(self, n_hid):
        super(Premodel,self).__init__()
        self.n_hid = n_hid
        self.dim_X = 4
        self.dim_input = self.dim_X + 1
        
        self.pre_Weight = nn.Parameter(torch.zeros(size=(self.dim_X, self.n_hid), requires_grad=True)).to(device)
        self.pre_Bias = nn.Parameter(torch.zeros(size=(1, self.n_hid), requires_grad=True)).to(device)
    
    def forward(self, X):
        if self.n_hid > self.dim_input:
            pad = torch.nn.ConstantPad1d((0, self.n_hid - self.dim_input), 0).to(device)
            output = pad(X)
        else:
            output = torch.matmul(X, self.pre_Weight) + self.pre_Bias
        return output.to(device)

#neural networks
class NeuralNetworks(nn.Module):
    def __init__(self, n_block, n_hid):
        super(NeuralNetworks, self).__init__()
        self.n_block = n_block
        self.n_hid = n_hid
        self.dim_X = 4
        self.dim_input = self.dim_X + 1

        self.Ritzblocks = nn.ModuleList().to(device)
        for _ in range(self.n_block): 
            model = RitzBlock(self.n_hid).to(device)
            self.Ritzblocks.append(model)
        
        self.pre_model = Premodel(self.n_hid).to(device)
        self.nn_Weight = nn.Parameter(torch.zeros(self.n_hid, self.dim_X), requires_grad=True).to(device)
        self.nn_Bias = nn.Parameter(torch.zeros(1, self.dim_X), requires_grad=True).to(device)

    def ActiveWork(self, gamma, traj, tlength, dt):
        dr = traj[1:, :, :] - traj[:-1, :, :]
        v_mean = 1/2 * (traj[:-1, :, :] + traj[1:, :, :])
        aw = gamma * torch.sum(v_mean * dr, (0, -1))
        return torch.mean(aw)/(dt*tlength)

    def GirsanovWeight(self, Diffusion, Force, tlength, dt):
        DT = Diffusion[0]; DR_prime = Diffusion[-1]
        pos = dt/DT * torch.sum(Force[:-1, :, :2]**2, (0, -1))
        vel = dt/DR_prime * torch.sum(Force[:-1, :, 2:]**2, (0, -1))
        Gir = 1/2 * (pos + vel)
        return torch.mean(Gir)/(dt*tlength)

    def Loss(self, lamb, gamma, Diffusion, traj, Force, tlength, dt):
        AW = self.ActiveWork(gamma, traj, tlength, dt)
        Gir = self.GirsanovWeight(Diffusion, Force, tlength, dt)
        return -lamb*AW + Gir

    def forward(self, X, lamb): 
        size = list(X[...,0].shape); size.append(1)
        lamb = lamb * torch.ones(size=size).to(device)
        XX = torch.cat([X, lamb], -1).to(device) 
        o = self.pre_model(XX).to(device) 
        
        for model in self.Ritzblocks:
            oo = model(o).to(device)
            o = oo.to(device)        
        O = torch.matmul(oo, self.nn_Weight) + self.nn_Bias 
        return O.to(device)

#ODE
class ODE(nn.Module):
    def __init__(self, nptl=1):
        super().__init__()
        self.nptl = nptl
    
    def F(self, spring_constant, Friction, X):
        gamma=Friction[0]; gammaR=Friction[1]
        r = X[:, :2]; v = X[:, 2:]
        
        f_r = v - spring_constant/gamma * r
        f_v = -gammaR * v
        return torch.cat([f_r, f_v], -1).to(device)
         
    def forward(self, spring_constant, Friction, X):        
        return self.F(spring_constant, Friction, X).to(device)

#SDE
class SDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"
    
    def __init__(self, spring_constant, Friction, Diffusion_matrix, device = device):
        super(SDE, self).__init__()
        self.dim_X = 4
        self.dim_input = self.dim_X + 1
        self.ode = ODE().to(device)
        self.k = spring_constant 
        self.motility = Friction 
        self.DMatrix = Diffusion_matrix.to(device)
                
    def Drift(self, t, X):
        F0 = self.ode(self.k, self.motility, X).to(device) 
        FNN = NN(X, lamb).to(device)
        return F0 + FNN 
        
    def Diffusion(self, t, X): 
        return self.DMatrix.expand(X.shape[0], self.dim_X).to(device)

#AOUP simulation function
def AOUP(Lambda, NN, n_epochs, lr, ntraj, spring_constant, Friction, Diffusion, tlength, dt):
    optimizer = optim.SGD(NN.parameters(), lr, momentum = 0.1) 
    NN.train()
    
    sde = SDE(spring_constant, Friction, Diffusion).to(device) #define stochatic differential equation
    if sde.sde_type == "ito": sde_method = "euler"
    else: sde_method = "midpoint"
    
    Time = torch.arange(0, dt*(1+tlength), dt).to(device)
    initial_X = torch.zeros(size=(ntraj, 4)).to(device) #initial condition
    
    loss, aw, gir = [], [], []
    
    for _ in tqdm(range(n_epochs), desc="epoch"):
        start = time.time()
        optimizer.zero_grad()
        with torch.no_grad():
            bm = BrownianInterval(t0=Time[0], t1=Time[-1], dt = dt,size=(ntraj, 4), device = device)
            traj = sdeint(sde, initial_X, Time, dt=dt, bm=bm, method=sde_method, 
                          names={"drift" : "Drift", "diffusion" : "Diffusion"}).to(device)
        
        F = NN.forward(traj, Lambda).to(device) 
        train_loss = NN.Loss(Lambda, Friction[0], Diffusion, traj, F, tlength, dt)
        #update parameters (backpropagation)
        
        train_loss.backward()
        optimizer.step()
        
        #update initial condition
        initial_X = traj[-1, :, :].detach().to(device)

        loss.append(train_loss.detach().item())
        aw.append(NN.ActiveWork(Friction[0], traj, tlength, dt).detach().item())
        gir.append(NN.GirsanovWeight(Diffusion, F, tlength, dt).detach().item())
           
        end=time.time()
        #print("Time : {0:.2f} sec, Train loss : {1:.2f}".format(end-start, train_loss.detach().item()))
    NN=NN.cpu()
    return loss, aw, gir

#set paramters
n_block = 2; n_hid = 200

kbT = 0.05 #temperature of the reservoir
d = 2; sigma = 1; Fa = 20; 
gamma = 10

DT = kbT/gamma; DR = 3 * DT/sigma
gamma_R = (d-1)*DR
DR_prime = (d-1)/d * DR * (Fa/gamma)**2

Friction = [gamma, gamma_R]

spring_constant = 1

Diffusion = torch.zeros(4)
for i in range(2): Diffusion[i]=np.sqrt(2*DT); Diffusion[i+2]=np.sqrt(2*DR_prime)

lamb = -10
dt=0.001; tlength=3000; lr=0.01
ntraj=1000; n_epochs=5000

NN=NeuralNetworks(n_block, n_hid)
loss, work, weight = AOUP(lamb, NN, n_epochs, lr, ntraj, spring_constant, Friction, Diffusion, tlength, dt)

loss_name="{0}_lambda_loss.pkl".format(lamb)
work_name="{0}_lambda_aw.pkl".format(lamb)
weight_name="{0}_lambda_gw.pkl".format(lamb)

with open(loss_name, "wb") as fa: pickle.dump(loss, fa)
with open(work_name, "wb") as fb: pickle.dump(work, fb)
with open(weight_name, "wb") as fc: pickle.dump(weight, fc)