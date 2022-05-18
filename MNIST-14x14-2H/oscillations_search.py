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


def RhoCloseToOne(rho,l,over,under,beta,gamma,tol = 10**-2):
    rho = abs(rho)
    if abs(1-rho) < tol:
        l.append((beta,gamma))
    elif rho > 1 :
        over.append((beta,gamma))
    else:
        under.append((beta,gamma))

for alpha in [0.01,0.05,0.1,0.25]:

    betaR = list(np.arange(0,1,0.005))[1:]
    gammaR = list(np.arange(0,1,0.005))[1:]

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

    good_params = []
    over_one = []
    under_one = []

    for beta in betaR:
        for i, gamma in enumerate(gammaR):
            if beta + gamma > 1:
                pass
            else:
                d = 196

                A11 = (1-beta-gamma) * np.eye(d)
                A12 = beta * Wba
                A21 = (1-beta-gamma) * gamma * Wab + alpha/d * Wba.T
                A22 = gamma * beta * Wab.dot(Wba) + (1-gamma) * np.eye(d) - alpha/d * Wba.T.dot(Wba)
                A = np.block([[A11,A12],[A21,A22]])

                rho,_ = eigs(A,k=1,which='LM',tol=10**-3) # ie spectral radius
                RhoCloseToOne(rho,good_params,over_one,under_one,beta,gamma,tol = 10**-2)


    np.save(os.path.join('oscillations_parameters_setup',f'good_params_{alpha}.npy'),good_params)
    np.save(os.path.join('oscillations_parameters_setup',f'over_params_{alpha}.npy'),over_one)
    np.save(os.path.join('oscillations_parameters_setup',f'under_params_{alpha}.npy'),under_one)


    good_params = np.load(os.path.join('oscillations_parameters_setup',f'good_params_{alpha}.npy'))
    over_one = np.load(os.path.join('oscillations_parameters_setup',f'over_params_{alpha}.npy'))
    under_one =  np.load(os.path.join('oscillations_parameters_setup',f'under_params_{alpha}.npy'))
    plt.figure()
    plt.scatter(*zip(*good_params),color='red',label='rho = 1')
    plt.scatter(*zip(*over_one),color='blue',label='rho > 1')
    plt.scatter(*zip(*under_one),color='green',label='rho < 1')
    plt.scatter(*zip(*good_params),color='red',label='rho = 1')
    x = np.linspace(0,1,100)
    plt.plot(x,1-x,linestyle='dashed',label='beta+gamma = 1',color='red')
    plt.xlabel('beta')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.ylabel('gamma')
    plt.title(f'Potential oscillations for alpha = {alpha}')
    plt.legend()
    plt.savefig(os.path.join('oscillations_parameters_setup',f'potential_good_parameters_{alpha}.png'))
    plt.close()

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
            if np.isreal(eig) or abs(1-abs(eig)) >= 10**-2 : 
                pass
            else:
                #print(eig)
                osci_eigv.append((beta,gamma,v[i]))
    inv = np.linalg.inv(Win)

    osci_imgs = [(beta,gamma,inv.dot(np.real(y[:196])-Winb)) for beta,gamma,y in osci_eigv]

    unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
    with open(os.path.join('oscillations_parameters_setup',f'params_dictionary_{alpha}.pkl'), 'wb') as f:
        pickle.dump(unflattened_imgs, f)

    unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
    print(len(unflattened_imgs_list))
    for i,img in enumerate(unflattened_imgs_list):
        _,_,img = img
        plt.imsave(os.path.join('oscillations_parameters_setup',f'img_{alpha}_{i}.png'),img, cmap='gray')
