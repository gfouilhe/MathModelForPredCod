import torch
from model import PCMLP
import os
import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

alpha = 0.01
beta = 0.5
gamma = 0.2
model= PCMLP(5,5,0,alpha,beta,gamma)
checkpointPhase = torch.load(os.path.join('models',f"PC_E29_I9.pth"))
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    if name=='fcAB.weight':
        Wab = p.detach().numpy()
    if name=='fcBA.weight':
        Wba = p.detach().numpy()


A11 = (1-beta-gamma) * np.eye(5)
A12 = beta * Wba
A21 = (1-beta-gamma) * gamma * Wab + alpha/5 * Wba.T
A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(5) - alpha/5 * Wba.T.dot(Wba)
A = np.block([[A11,A12],[A21,A22]])

w, v = np.linalg.eig(A)

print(abs(w))

# all eigenvalues modules are < 1 !


betaR = list(np.arange(0,1,0.01))[1:]
gammaR = list(np.arange(0,1,0.01))[1:]
alphaR = list(np.arange(0,1,0.01))[1:]

for beta in betaR:
    for gamma in gammaR:
        for alpha in alphaR:
            if gamma + alpha + alpha < 1 :
                
                A11 = (1-beta-gamma) * np.eye(5)
                A11 = (1-beta-gamma) * np.eye(5)
                A12 = beta * Wba
                A21 = (1-beta-gamma) * gamma * Wab + alpha/5 * Wba.T
                A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(5) - alpha/5 * Wba.T.dot(Wba)
                A = np.block([[A11,A12],[A21,A22]])

                w, v = np.linalg.eig(A)
                if max(abs(w))==1:
                    print('=1',beta,gamma,alpha, alpha+beta+gamma)
                if max(abs(w))>1:
                    print('>1',beta,gamma,alpha, alpha+beta+gamma)

# For this learned weights, any parameters choice beta, gamma, alpha in (0,1] give the same result.

