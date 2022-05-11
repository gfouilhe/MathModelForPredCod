import torch
from model import PCMLP
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def ClosestToOne(w):
    inf = w[0]
    iinf = 0
    sup = w[0]
    isup = 0
    ones = []

    for i,eig in enumerate(w):
        if abs(eig) < 1:
            if abs(eig) > abs(inf):
                inf = eig
                iinf = i
        elif abs(eig) > 1:
            if abs(eig)< abs(sup):
                sup = eig
                isup = i
        else:
            ones.append((i,eig))
    return ones, inf, iinf, sup, isup


alpha = 0.01

betaR = list(np.arange(0,1,0.01))[1:]
gammaR = list(np.arange(0,1,0.01))[1:]

threshold = []

model= PCMLP(0.33,alpha,beta,gamma)
checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I4.pth"))
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    if name=='fcAB.weight':
        Wab = p.detach().numpy()
    if name=='fcBA.weight':
        Wba = p.detach().numpy()


for beta in betaR:
    for i, gamma in enumerate(gammaR):
        if beta + gamma > 1:
            break
        d = 120

        A11 = (1-beta-gamma) * np.eye(d)
        A12 = beta * Wba
        A21 = (1-beta-gamma) * gamma * Wab + alpha/d * Wba.T
        A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
        A = np.block([[A11,A12],[A21,A22]])

        w, v = np.linalg.eig(A)
        ones, inf, iinf, sup, isup = ClosestToOne(w)

        if i==0:
            liminf = inf
            iliminf = iinf
            limsup = sup
            ilimsup = ilimsup
            ones = ones



                