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

gamma_beta_couples = [(0.1,0.7),(0.25,0.5),(0.33,0.33),(0.5,0.25),(0.7,0.1)]
alpha_range = [0.01]#,0.05,0.1,0.5]
it = 0
for gamma, beta in gamma_beta_couples:
    for alpha in alpha_range:
        it+=1
        if it%5==0:
            print("iteration : ",it)

        betaR = list(np.arange(0,1,0.01))[1:]
        gammaR = list(np.arange(0,1,0.01))[1:]

        model= PCMLP(0.33,alpha,beta,gamma)
        
        checkpointPhase = torch.load(os.path.join('models',f"PCT_E19_I0_G{gamma}_B{beta}_A{alpha}.pth"))
        print(gamma,beta,alpha)
        model.load_state_dict(checkpointPhase["module"])
        for name, p in model.named_parameters():
            tmp = p.detach().numpy()
            if name=='fcAB.weight':
                W12 = p.detach().numpy()
            if name=='fcBA.weight':
                W21 = p.detach().numpy()
            if name=='fciA.weight':
                W01 = p.detach().numpy()
            if name=='fcAi.weight':
                W10 = p.detach().numpy()

        good_params = []
        over_one = []
        under_one = []
        d = W12.shape[1]

        for betab in betaR:
            for i, gammab in enumerate(gammaR):
                if betab + gammab > 1:
                    pass
                else:
                

                    A11 = (1-betab-gammab) * np.eye(d)
                    A12 = betab * W21
                    A21 = (1-betab-gammab) * gammab * W12 + alpha/d * W21.T
                    A22 = gammab * betab * W12.dot(W21) + (1-gammab) * np.eye(d) - alpha/d * W21.T.dot(W21)

                    A = np.block([[A11,A12],[A21,A22]])

                    rho,_ = eigs(A,k=1,which='LM',tol=10**-3) # ie spectral radius
                    RhoCloseToOne(rho,good_params,over_one,under_one,betab,gammab,tol = 10**-3)

        print(gamma,beta,alpha)
        np.save(os.path.join('oscillations_parameters_setup',f'Tgood_params_G{gamma}_B{beta}_A{alpha}.npy'),good_params)
        np.save(os.path.join('oscillations_parameters_setup',f'Tover_params_G{gamma}_B{beta}_A{alpha}.npy'),over_one)
        np.save(os.path.join('oscillations_parameters_setup',f'Tunder_params_G{gamma}_B{beta}_A{alpha}.npy'),under_one)
        

        good_params = np.load(os.path.join('oscillations_parameters_setup',f'Tgood_params_G{gamma}_B{beta}_A{alpha}.npy'))
        over_one = np.load(os.path.join('oscillations_parameters_setup',f'Tover_params_G{gamma}_B{beta}_A{alpha}.npy'))
        under_one =  np.load(os.path.join('oscillations_parameters_setup',f'Tunder_params_G{gamma}_B{beta}_A{alpha}.npy'))
        plt.figure()
        plt.scatter(*zip(*over_one),color='blue',label='$\\rho > 1$')
        plt.scatter(*zip(*under_one),color='green',label='$\\rho < 1$')
        
        if not len(good_params) == 0:
            plt.scatter(*zip(*good_params),color='red',label='$\\rho = 1$')
        x = np.linspace(0,1,100)
        plt.plot(x,1-x,linestyle='dashed',label='$\lambda+\\beta = 1$',color='red')
        plt.scatter(beta,gamma,marker='x',label='Hyperparameters used for training')
        plt.xlabel('$\lambda$')
        plt.xlim((0,1))
        plt.ylim((0,1))
        plt.ylabel('$\\beta$')
        plt.title(f'Potential oscillations for $\\alpha = {alpha}$')
        plt.legend()
        plt.savefig(os.path.join('oscillations_parameters_setup',f'Tpotential_good_parameters_G{gamma}_B{beta}_A{alpha}.png'))
        plt.close()

        osci_eigv = []
        for betab,gammab in good_params:
            

            A11 = (1-betab-gammab) * np.eye(d)
            A12 = betab * W21
            A21 = (1-betab-gammab) * gammab * W12 + alpha/d * W21.T
            A22 = gammab * betab * W12.dot(W21) + (1-gammab) * np.eye(d) - alpha/d * W21.T.dot(W21)
            A = np.block([[A11,A12],[A21,A22]])

            w, v = np.linalg.eig(A)
            for i,eig in enumerate(w):
                if np.isreal(eig) or abs(1-abs(eig)) >= 10**-3 : 
                    pass
                else:
                    #print(eig)
                    osci_eigv.append((betab,gammab,v[i]))

        osci_imgs = [(betab,gammab,np.real(y[:196])) for betab,gammab,y in osci_eigv]

        unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
        with open(os.path.join('oscillations_parameters_setup',f'Tparams_dictionary_G{gamma}_B{beta}_A{alpha}.pkl'), 'wb') as f:
            pickle.dump(unflattened_imgs, f)

        unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
        print("Number of generated images :", len(unflattened_imgs_list))
        # for i,img in enumerate(unflattened_imgs_list):
        #     _,_,img = img
        #     plt.imsave(os.path.join('oscillations_parameters_setup',f'img_{alpha}_{i}.png'),img, cmap='gray')
        print(gamma,beta,alpha)