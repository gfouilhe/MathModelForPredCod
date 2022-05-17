from sympy import unflatten
import torch
from model import PCMLP
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
from scipy.sparse.linalg import eigs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def RhoCloseToOne(rho,l,beta,gamma,tol = 10**-4):
    if abs(1-rho) < tol:
        return l.append((beta,gamma))


alpha = 0.01

betaR = list(np.arange(0,1,0.005))[1:]
gammaR = list(np.arange(0,1,0.005))[1:]

threshold = []

model= PCMLP(0.33,alpha,0.5,0.2)
checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0.pth"))
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
    if name=='fcBC.weight':
        Wbc = tmp
    if name=='fcCB.weight':
        Wcb = tmp
    if name=='fcCD.weight':
        Wcd = tmp
    if name=='fcDC.weight':
        Wdc = tmp

# good_params = []
# for beta in betaR:
#     for i, gamma in enumerate(gammaR):
#         if beta + gamma > 1:
#             pass
#         else:
#             d = 196

#             A11 = (1-beta-gamma) * np.eye(d)
#             A12 = beta * Wba
#             A13 = np.zeros((d,d))
#             A21 = (1-beta-gamma) * gamma * Wba.T + alpha/d * Wba.T
#             A22 = gamma * beta * Wab.dot(Wba) + (1-beta-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
#             A23 = beta * Wcb
#             A31 = (1-beta-gamma) * gamma**2 * Wbc.dot(Wab) + alpha/d * gamma * Wbc.dot(Wcb.T)
#             A32 = beta * gamma **2 * Wab.dot(Wba) + (1-beta-gamma) * gamma * Wbc - alpha/d * gamma * Wbc.dot(Wba.T.dot(Wba)) + alpha/d * Wcb.T
#             A33 = beta * gamma * Wbc * Wcb + (1-gamma) * np.eye(d) - alpha/d * Wcb.T.dot(Wcb)
#             A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])

#             w,_ = eigs(A,k=1,which='LM',tol=10**-5) # ie spectral radius
#             RhoCloseToOne(w,good_params,beta,gamma,tol = 10**-3)


# np.save(os.path.join('oscillations_parameters_setup','good_params.npy'),good_params)


good_params = np.load(os.path.join('oscillations_parameters_setup','good_params.npy'))  
plt.figure()
plt.scatter(*zip(*good_params))
x = np.linspace(0,1,100)
plt.plot(x,1-x,linestyle='dashed',label='beta+gamma = 1',color='red')
plt.xlabel('beta')
plt.xlim((0,1))
plt.ylim((0,1))
plt.ylabel('gamma')
plt.title('Potential oscillations for alpha = 0.01')
plt.legend()
plt.savefig(os.path.join('oscillations_parameters_setup','potential_good_parameters.png'))
plt.show()

d = 196
osci_eigv = []
for beta,gamma in good_params:
    

    A11 = (1-beta-gamma) * np.eye(d)
    A12 = beta * Wba
    A21 = (1-beta-gamma) * gamma * Wab + alpha/d * Wba.T
    A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
    A = np.block([[A11,A12],[A21,A22]])

    w, v = np.linalg.eig(A)
    for i,eig in enumerate(w):
        if np.isreal(eig) or abs(1-abs(eig)) > 10**-3 :
            pass
        else:
            osci_eigv.append((beta,gamma,v[i]))
inv = np.linalg.inv(Win)

osci_imgs = [(beta,gamma,inv.dot(y[:196].astype('float64')-Winb)) for beta,gamma,y in osci_eigv]

unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
with open(os.path.join('oscillations_parameters_setup','params_dictionary.pkl'), 'wb') as f:
    pickle.dump(unflattened_imgs, f)

unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
print(len(unflattened_imgs_list))
for i,img in enumerate(unflattened_imgs_list):
    _,_,img = img
    plt.imsave(os.path.join('oscillations_parameters_setup',f'img{i}.png'),img, cmap='gray')


