from sympy import unflatten
import torch
from model import PCMLP
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def ClosestToOne(w,tol = 10**-4):
    inf = w[0]
    iinf = 0
    sup = w[0]
    isup = 0
    ones = []

    for i,eig in enumerate(w):
        if abs(eig) < 1 - tol:
            if abs(eig) > abs(inf):
                inf = eig
                iinf = i
        elif abs(eig) > 1 + tol:
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

model= PCMLP(0.33,alpha,0.5,0.2)
checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I4.pth"))
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    tmp = p.detach().numpy()
    if name=='fcAB.weight':
        Wab = tmp
    if name=='fcBA.weight':
        Wba = tmp
    if name=='fcin.weight':
        Win = tmp
    if name=='fcin.bias':
        Winb = tmp
'''
good_params = []
for beta in betaR:
    for i, gamma in enumerate(gammaR):
        if beta + gamma > 1:
            pass
        else:
            d = 120

            A11 = (1-beta-gamma) * np.eye(d)
            A12 = beta * Wba
            A21 = (1-beta-gamma) * gamma * Wab + alpha/d * Wba.T
            A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
            A = np.block([[A11,A12],[A21,A22]])

            w, v = np.linalg.eig(A)
            ones, inf, iinf, sup, isup = ClosestToOne(w)

            if ones:
                good_params.append((beta,gamma))


np.save(os.path.join('oscillations_parameters_setup','good_params.npy'),good_params)
'''

good_params = np.load(os.path.join('oscillations_parameters_setup','good_params.npy'))
plt.figure()
plt.scatter(*zip(*good_params))
plt.xlabel('beta')
plt.ylabel('gamma')
plt.title('Potential oscillations for alpha = 0.01')
plt.savefig(os.path.join('oscillations_parameters_setup','potential_good_parameters.png'))

d = 120
osci_eigv = []
for beta,gamma in good_params:
    

    A11 = (1-beta-gamma) * np.eye(d)
    A12 = beta * Wba
    A21 = (1-beta-gamma) * gamma * Wab + alpha/d * Wba.T
    A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
    A = np.block([[A11,A12],[A21,A22]])

    w, v = np.linalg.eig(A)
    for i,eig in enumerate(w):
        if np.isreal(eig):
            pass
        else:
            osci_eigv.append((beta,gamma,v[i]))
pseudo_inv = np.linalg.pinv(Win)

osci_imgs = [(beta,gamma,pseudo_inv.dot(y[:120].astype('float64')-Winb)) for beta,gamma,y in osci_eigv]

unflattened_imgs = [img.reshape((28,28)) for _,_,img in osci_imgs]

for i,img in enumerate(unflattened_imgs):
    if i < 3014:
        pass
    else:
        plt.figure()
        plt.imshow(img,cmap='gray')
        plt.savefig(os.path.join('oscillations_parameters_setup',f'img{i}.png'))
        plt.close()


