import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings(action="ignore")
from torchsde import BrownianInterval, sdeint
import pickle

CUDA = torch.cuda.is_available()
CUDA_DEVICE = 0
if CUDA: device="cuda:0"
else: device="cpu"

#Ritz Blocks
class RitzBlock(nn.Module):
    def __init__(self, n_hid):
        super(RitzBlock, self).__init__()
        self.n_hid = n_hid
        self.activation=nn.Tanh()

        ##Layer 1
        self.Weight_layer1 = nn.Parameter(torch.zeros(size=(n_hid,n_hid),requires_grad=True).to(device))
        self.Bias_layer1 = nn.Parameter(torch.zeros(size=(1,n_hid),requires_grad=True).to(device))    
        ##Layer 2
        self.Weight_layer2 = nn.Parameter(torch.zeros(size=(n_hid,n_hid),requires_grad=True).to(device))
        self.Bias_layer2 = nn.Parameter(torch.zeros(size=(1,n_hid),requires_grad=True).to(device))

    def layer1(self,X): 
        return self.activation(torch.matmul(X, self.Weight_layer1)+self.Bias_layer1).to(device)
    def layer2(self,X): 
        return self.activation(torch.matmul(X, self.Weight_layer2)+self.Bias_layer2).to(device)
    
    def forward(self,X):
        h1 = self.layer1(X).to(device)
        h2 = self.layer2(h1).to(device)
        return h2 + X

#pre model
class PreModel(nn.Module):
    def __init__(self, n_hid, nptl):
        super(PreModel, self).__init__()
        self.n_hid= n_hid
        self.nptl = nptl
        self.dim_X = 4*nptl
        self.dim_input = self.dim_X + 1
        self.pre_Weight = nn.Parameter(torch.zeros(size=(self.dim_X, self.n_hid), requires_grad = True).to(device))
        self.pre_Bias = nn.Parameter(torch.zeros(size=(1, self.dim_X), requires_grad = True).to(device))
        
    def forward(self, X):
        if self.n_hid > self.dim_input:
            pad = torch.nn.ConstantPad1d((0, self.n_hid - self.dim_input), 0)
            output = pad(X)
        else:
            output = torch.matmul(X, self.pre_Weight) + self.pre_Bias
        return output.to(device)
        
#neural networks
class NeuralNetworks(nn.Module):
    def __init__(self, n_block, n_hid, nptl):
        super(NeuralNetworks, self).__init__()
        self.n_blocks = n_block
        self.n_hid = n_hid
        self.nptl = nptl
        self.dim_X = 4*self.nptl
        self.dim_input = self.dim_X + 1
        
        self.Ritzblocks = nn.ModuleList()
        for _ in range(self.n_blocks): self.Ritzblocks.append(RitzBlock(self.n_hid).to(device))
        self.pre_model = PreModel(self.n_hid, self.nptl).to(device)
        self.nn_Weight = nn.Parameter(torch.zeros(size=(self.n_hid, self.dim_X), requires_grad = True).to(device))
        self.nn_Bias = nn.Parameter(torch.zeros(size = (1, self.dim_X), requires_grad = True).to(device))
        
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

#define loss function
def ActiveWork(k, Motilities, Diffusion, traj, Force, noise, tlength, dt):
    gamma = Motilities[0]; DT = Diffusion[0]
    nptl = int(traj.shape[-1]/4)
    Noise = torch.zeros(size=(tlength, traj.shape[1], traj.shape[-1])).to(device)
    for t in range(tlength): Noise[t, ...] = noise(t*dt, (t+1)*dt)
    
    first = torch.mean(torch.sum(traj[:-1, :, 2*nptl:]**2, (0, -1)))
    second = k/gamma * torch.mean(torch.sum(traj[:-1, :, :2*nptl]*traj[:-1, :, 2*nptl:], (0, -1)))
    third = torch.mean(torch.sum(traj[:-1, :, 2*nptl:]*Force[:-1, :, :2*nptl], (0, -1)))
    fourth = torch.mean(torch.sum((traj[:-1, :, 2*nptl:]+traj[1:, :, 2*nptl:])/2 * Noise[:, :,:2*nptl], (0, -1)))
    
    aw = gamma * (first - second + third + fourth)
    return aw*dt/(dt*tlength) #time averaged quantity
    
def GirsanovWeight(Diffusion, Force, tlength, dt):
    nptl = int(Force.shape[-1]/4)
    DT = Diffusion[0]; DR = Diffusion[-1]
    pos = 1/DT*torch.mean(torch.sum(Force[:-1, :, :2*nptl]**2, (0, -1)))
    vel = 1/DR*torch.mean(torch.sum(Force[:-1, :, 2*nptl:]**2, (0, -1)))
    Gir = pos + vel
    return Gir*dt/(2*dt*tlength)

#ODE
class ODE(nn.Module):
    def __init__(self, nptl):
        super().__init__()
        self.nptl = nptl
    
    def F(self, spring_constant, Motilities, X):
        gamma=Motilities[0]; gammaR=Motilities[1]
        f_r = X[..., 2*self.nptl:] - spring_constant/gamma * X[..., :2*self.nptl]
        f_v = -gammaR * X[..., 2*self.nptl:]
        return torch.cat([f_r, f_v], -1).to(device)
         
    def forward(self, spring_constant, Motilities, X):        
        return self.F(spring_constant, Motilities, X).to(device)

#SDE
class SDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"
    
    def __init__(self, nptl, spring_constant, Motilities, Diffusion_matrix, device = device):
        super(SDE, self).__init__()
        self.nptl = nptl
        self.dim_X = 4*nptl
        self.dim_input = self.dim_X + 1
        self.ode = ODE(nptl).to(device)
        self.k = spring_constant 
        self.motility = Motilities 
        self.DMatrix = Diffusion_matrix.to(device)
        
    def Drift(self, t, X):
        F0 = self.ode(self.k, self.motility, X).to(device) 
        FNN = NN(X, Lambda).to(device)
        return F0 + FNN 
        
    def Diffusion(self, t, X): 
        return self.DMatrix.expand(X.shape[0], self.dim_X).to(device)

#AOUP simulation function
def AOUP(Lambda, NNs, n_epochs, lr, sample_size, nptl, spring_constant, Motilities, Diffusion, tlength, dt):
    optimizer = optim.SGD(NNs.parameters(), lr, momentum = 0.9) 
    NNs.train()
    
    initial_X = torch.zeros(size=(sample_size, 4*nptl)).to(device) #initial condition
    sde = SDE(nptl, spring_constant, Motilities, Diffusion).to(device) #define stochatic differential equation
    if sde.sde_type == "ito": sde_method = "euler"
    else: sde_method = "midpoint"
    time = torch.arange(0, dt*tlength + dt, dt).to(device)
    
    losses = []; Observables =[]; Girsanovs = [] 
    
    for _ in tqdm(range(n_epochs), desc="epoch"):
        with torch.no_grad():
            noise = BrownianInterval(t0=time[0], t1=time[-1], size=(sample_size, 4*nptl), device = device)
            traj = sdeint(sde, initial_X, time, dt = dt, bm = noise, method = sde_method, 
                          names = {"drift" : "Drift", "diffusion" : "Diffusion"}).to(device)
        
        F = NNs.forward(traj, Lambda).to(device) 
        
        #calculate train loss
        AW = ActiveWork(spring_constant, Motilities, Diffusion, traj, F, noise, tlength, dt)
        GW = GirsanovWeight(Diffusion, F, tlength, dt)
        train_loss = -Lambda * AW + GW

        #update parameters (backpropagation)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        #update initial condition
        initial_X = traj[-1, :, :].detach().to(device)

        losses.append(train_loss.detach().item())
        Observables.append(AW.detach().item())
        Girsanovs.append(GW.detach().item())

    NNs=NNs.cpu()
    return losses, Observables, Girsanovs

#set parameters
Lambdas = [-10, -9, -8]
n_hid = 150; n_blocks = 3; nptl = 1; 

DT = 1; DR = 1
gamma = 1; gammaR = 1; Motilities = [gamma, gammaR]

spring_constant = 2

Diffuse = torch.ones(4*nptl).to(device) 
for i in range(2*nptl): Diffuse[i] = np.sqrt(2*DT) ; Diffuse[i+2*nptl] = np.sqrt(2*DR)

sample_size = 500 #number of trajectories
dt = 1e-3; tlength = 1500
lr = 0.001; n_epoch = 3000

loss_dict = {}
work_dict = {}
weight_dict = {}

for Lambda in Lambdas:
    loss_dict[Lambda] = 0
    work_dict[Lambda] = 0
    weight_dict[Lambda] = 0

for Lambda in Lambdas:
    NN = NeuralNetworks(n_blocks, n_hid, nptl).to(device)
    losses, work, weight = AOUP(Lambda, NN, n_epoch, lr, sample_size, nptl, spring_constant, Motilities, Diffuse, tlength, dt)
    loss_dict[Lambda] = losses; work_dict[Lambda] = work; weight_dict[Lambda] = weight
    
loss_name = "{0}_to_{1}_loss.pkl".format(Lambdas[0], Lambdas[-1])
work_name = "{0}_to_{1}_AW.pkl".format(Lambdas[0], Lambdas[-1])
weight_name = "{0}_to_{1}_GW.pkl".format(Lambdas[0], Lambdas[-1])

with open(loss_name, "wb") as fa: pickle.dump(loss_dict, fa)
with open(work_name, "wb") as fb: pickle.dump(work_dict, fb)
with open(weight_name, "wb") as fc: pickle.dump(weight_dict, fc)