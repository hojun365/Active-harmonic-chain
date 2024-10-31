import numpy as np

def PrimaryDomain(k:int,N:int,tau:float,mu:float,K:float,Da:float,kbT:float):
    tau_k = 1/tau+4*mu*K*np.sin(np.pi*k/N)**2; coeff = kbT*mu*tau**2/Da
    left = -1/(2*kbT) - 1/(2*kbT)*np.sqrt(1+coeff*tau_k**2); right = -1/(2*kbT) + 1/(2*kbT)*np.sqrt(1+coeff*tau_k**2)
    return left, right
    
def CGF_k(k:int,N:int,tau:float,mu:float,K:float,Da:float,kbT:float,lmbda):
    tau_k = 1/tau+4*mu*K*np.sin(np.pi*k/N)**2; coeff = 4*Da/(kbT*mu*tau**2)
    return tau_k/2 - np.sqrt(tau_k**2-coeff*(kbT*lmbda)*(kbT*lmbda+1))/2

def AW_k(k:int,N:int,tau:float,mu:float,K:float,Da:float,kbT:float,lmbda):
    coeff = Da/(mu*tau**2)
    a = 4*Da/(kbT*mu*tau**2); b = 1/tau+4*mu*K*np.sin(np.pi*k/N)**2
    return coeff*(2*kbT*lmbda+1)*1/np.sqrt(b**2-a*(kbT*lmbda)*(kbT*lmbda+1))

def CGF(N:int,tau:float,mu:float,K:float,Da:float,kbT:float,lmbda):
    cgf = 0
    for k in range(N): cgf += CGF_k(k,N,tau,mu,K,Da,kbT,lmbda)
    return cgf

def AW(N:int,tau:float,mu:float,K:float,Da:float,kbT:float,lmbda):
    AW = 0
    for k in range(N): AW += AW_k(k,N,tau,mu,K,Da,kbT,lmbda)
    return AW