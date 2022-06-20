
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

for alpha in [0.01]: #,0.05,0.1,0.25]:

    betaR = list(np.arange(0,1,0.005))[1:]
    gammaR = list(np.arange(0,1,0.005))[1:]

    threshold = []

    model= PCMLP(0.33,alpha,0.7,0.1)
    checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0_G0.7_B0.1_A0.01.pth"))
    model.load_state_dict(checkpointPhase["module"])
    for name, p in model.named_parameters():
        tmp = p.detach().numpy()
        # if name=='fcAB.weight':
        #     W12 = tmp
        # if name=='fcBA.weight':
        #     W21 = tmp
        # if name=='fciA.weight':
        #     W01 = tmp
        # if name=='fcAi.weight':
        #     W10 = tmp

        if name=='fciA.fc_r.weight':
            W01R = tmp
        if name=='fciA.fc_i.weight':
            W01I = tmp
        if name=='fcAB.fc_r.weight':
            W12R = tmp
        if name=='fcAB.fc_i.weight':
            W12I = tmp
        if name=='fcAi.fc_r.weight':
            W10R = tmp
        if name=='fcAi.fc_i.weight':
            W10I = tmp
        if name=='fcBA.fc_r.weight':
            W21R = tmp
        if name=='fcBA.fc_i.weight':
            W21I = tmp

    d = W01R.shape[1]
    Z = np.zeros((d,d))
    W01 = np.block([[W01R,Z],[Z,W01I]])
    W10 = np.block([[W10R,Z],[Z,W10I]])
    W12 = np.block([[W12R,Z],[Z,W12I]])
    W21 = np.block([[W21R,Z],[Z,W21I]])
    good_params = []
    over_one = []
    under_one = []
    d = W01.shape[1]
    print(d)

    # for beta in betaR:
    #     for i, gamma in enumerate(gammaR):
    #         if beta + gamma > 1:
    #             pass
    #         else:
               

    #             A11 = (1-beta-gamma) * np.eye(d)
    #             A12 = beta * W21
    #             A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
    #             A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)

    #             A = np.block([[A11,A12],[A21,A22]])

    #             rho,_ = eigs(A,k=1,which='LM',tol=10**-3) # ie spectral radius
    #             RhoCloseToOne(rho,good_params,over_one,under_one,beta,gamma,tol = 10**-2)


    # np.save(os.path.join('oscillations_parameters_setup',f'Cgood_params_{alpha}.npy'),good_params)
    # np.save(os.path.join('oscillations_parameters_setup',f'Cover_params_{alpha}.npy'),over_one)
    # np.save(os.path.join('oscillations_parameters_setup',f'Cunder_params_{alpha}.npy'),under_one)


    good_params = np.load(os.path.join('oscillations_parameters_setup',f'Cgood_params_{alpha}.npy'))
    over_one = np.load(os.path.join('oscillations_parameters_setup',f'Cover_params_{alpha}.npy'))
    under_one =  np.load(os.path.join('oscillations_parameters_setup',f'Cunder_params_{alpha}.npy'))
    plt.figure()
    plt.scatter(*zip(*over_one),color='blue',label='rho > 1')
    plt.scatter(*zip(*under_one),color='green',label='rho < 1')
    plt.scatter(*zip(*good_params),color='red',label='rho = 1')
    x = np.linspace(0,1,100)
    plt.plot(x,1-x,linestyle='dashed',label='$\lambda+\\beta = 1$',color='red')
    plt.xlabel('$\lambda$')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.ylabel('$\\beta$')
    plt.title(f'Potential oscillations for $\\alpha = {alpha}$')
    plt.legend()
    plt.savefig(os.path.join('oscillations_parameters_setup',f'Cpotential_good_parameters_{alpha}.png'))
    plt.close()

    osci_eigv = []
    for beta,gamma in good_params:
        

        A11 = (1-beta-gamma) * np.eye(d)
        A12 = beta * W21
        A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
        A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
        A = np.block([[A11,A12],[A21,A22]])

        w, v = np.linalg.eig(A)
        for i,eig in enumerate(w):
            if np.isreal(eig) or abs(1-abs(eig)) >= 10**-2 :
                if np.isreal(eig):
                    print("real :'(")
        
            else:
                #print(eig)
                osci_eigv.append((beta,gamma,v[i]))

    osci_imgs = [(beta,gamma,y[:196]) for beta,gamma,y in osci_eigv]# np.real(y[:196])) for beta,gamma,y in osci_eigv]

    unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
    with open(os.path.join('oscillations_parameters_setup',f'Cparams_dictionary_{alpha}.pkl'), 'wb') as f:
        pickle.dump(unflattened_imgs, f)

    unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
    print(len(unflattened_imgs_list))
    # for i,img in enumerate(unflattened_imgs_list):
    #     _,_,img = img
    #     plt.imsave(os.path.join('oscillations_parameters_setup',f'ACimg_{alpha}_{i}.png'),img, cmap='gray')
