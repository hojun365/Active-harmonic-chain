import math
import torch
import torch.nn as nn
#-----------------------------------------------------------------------------------------------#
# Fully-connected
class Ritz(nn.Module):
    def __init__(self,n_hid):
        super(Ritz,self).__init__()
        self.activation = nn.Tanh()

        # neural network
        self.Layer1 = nn.Linear(n_hid,n_hid,bias=False)
        self.Layer2 = nn.Linear(n_hid,n_hid,bias=False)

        # Xavier initialization, gain=5/3 for tanh activation function
        nn.init.xavier_uniform_(self.Layer1.weight, gain=5/3)
        nn.init.xavier_uniform_(self.Layer2.weight, gain=5/3)

        # layer normalization
        self.ln = nn.LayerNorm(n_hid,elementwise_affine=False,bias=False)

        # model
        self._Ritz = nn.Sequential(*[self.Layer1, self.ln, self.activation,
                                     self.Layer2, self.ln, self.activation])

    def forward(self,X):
        return self._Ritz(X) + X
class FCN(nn.Module):
    def __init__(self,N,n_hid,n_block,bias=False):
        super(FCN,self).__init__()
        self.activation = nn.Tanh()

        # neural network
        if   bias == True:  self.encoder = nn.Linear(2*N+1,n_hid,bias=False)
        elif bias == False: self.encoder = nn.Linear(2*N+0,n_hid,bias=False)
        self.decoder = nn.Linear(n_hid,2*N,bias=False)

        # batch normalization (decoder에는 필요가 없다)
        self.ln = nn.LayerNorm(n_hid,elementwise_affine=False,bias=False)

        # xavier initialization
        nn.init.xavier_uniform_(self.encoder.weight, gain=5/3)
        nn.init.xavier_uniform_(self.decoder.weight, gain=1.0) # decoder의 경우 no activation

        # sequence construct
        self._Ritz = nn.Sequential(*[Ritz(n_hid) for _ in range(n_block)])

    def forward(self,X):
        X = self.activation(self.ln(self.encoder(X)))
        X = self._Ritz(X)
        X = self.decoder(X)
        return X
#-----------------------------------------------------------------------------------------------#
# Equivariant
class SAB(nn.Module):
    def __init__(self,N,dim_in,dim_att,dim_out,encoder=True,decoder=True):
        super(SAB,self).__init__()
        self.N = N
        self.dim_att = dim_att
        self.dim_out = dim_out
        self.encoder = encoder; self.decoder = decoder
        if   decoder == True:  self.activation = nn.LeakyReLU(negative_slope=1.0)
        elif decoder == False: self.activation = nn.Tanh()
        if encoder == True: self.first = nn.Linear(dim_in,dim_in,bias=False)
        self.ln0 = nn.LayerNorm(N*dim_att,elementwise_affine=False,bias=False)
        self.ln1 = nn.LayerNorm(N*dim_out,elementwise_affine=False,bias=False)
        # neural network
        self.Q = nn.Linear(dim_in,dim_att,bias=False); nn.init.xavier_uniform(self.Q.weight)
        self.K = nn.Linear(dim_in,dim_att,bias=False); nn.init.xavier_uniform(self.K.weight)
        self.V = nn.Linear(dim_in,dim_att,bias=False); nn.init.xavier_uniform(self.V.weight)
        self.Linear = nn.Linear(dim_att,dim_out,bias=False); nn.init.xavier_uniform(self.Linear.weight,gain=5/3)

    def forward(self,X):
        # X.shape = (...,2*N)
        if self.encoder == True: X = self.first(torch.cat([X[...,:self.N].unsqueeze(-1),X[...,self.N:2*self.N].unsqueeze(-1),X[...,2*self.N:].unsqueeze(-1)],-1)) # (...,N,3)
        else: X = X
        Q = self.Q(X).unsqueeze(-1); K = self.K(X).unsqueeze(-1); V = self.V(X)
        if self.decoder == True:
            W = torch.einsum("...nmh,...mh->...nh",torch.einsum("...nhi,...mhi->...nmh",Q,K),V)/self.N + V
            return self.Linear(W)
        else:
            W = self.ln0((torch.einsum("...nmh,...mh->...nh",torch.einsum("...nhi,...mhi->...nmh",Q,K),V)/self.N+V).reshape(*V.shape[:-2],self.N*self.dim_att)).view(*V.shape[:-2],self.N,self.dim_att)
            return self.activation(self.ln1(self.Linear(W).reshape(*W.shape[:-2],self.N*self.dim_out)).view(*W.shape[:-2],self.N,self.dim_out))

class Equiv(nn.Module):
    def __init__(self,N,n_hid,n_block):
        super(Equiv,self).__init__()
        # Parameters
        model = []
        model.append(SAB(N,3,n_hid//2,n_hid,encoder=True,decoder=False)) # encoder
        for _ in range(n_block): model.append(SAB(N,n_hid,n_hid//2,n_hid,encoder=False,decoder=False)) # Ritz block
        model.append(SAB(N,n_hid,4,2,encoder=False,decoder=True)) # decoder
        self._network = nn.Sequential(*model)

    def forward(self,X):
        X = self._network(X)
        return torch.cat([X[...,:1].squeeze(-1),X[...,1:].squeeze(-1)],-1)