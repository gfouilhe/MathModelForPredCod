
import torch
from model import PCMLP
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle 
from scipy.sparse.linalg import eigs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def RhoCloseToOne(rho,l,over,under,beta,gamma,tol1 = 10**-2,tolover = 0.5, tolunder = 0.2):
    rho = abs(rho)
    if abs(1-rho) < tol1:
        l.append((beta,gamma))
    elif rho > 1 + tolover :
        over.append((beta,gamma))
    elif rho < 1 - tolunder :
        under.append((beta,gamma))

batchSize = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
activation = torch.tanh

def main():
    
    #-------Parameters-----

    # These are the parameters used for computing eigenvalues, not those used to learn the weights.
    alphaR = [0.01]#,0.05,0.1,0.25]
    betaR = list(np.arange(0,1,0.01))[1:] 
    gammaR = list(np.arange(0,1,0.01))[1:]
    #

    # Parameters used for learning :
    UsedForLearningHyper = [(0.1,0.8,0.01),(0.1,0.5,0.01), (0.1,0.1,0.01),(0.33,0.33,0.01),(0.2,0.5,0.01),(0.5,0.2,0.01),(0.5,0.1,0.01),(0.8,0.1,0.01)]
    #(0.1,0.8,0.01),

    # Tolerences :
    tolOne = 0.0005
    tolOver = 0.01
    tolUnder = 0.01

    # Others :
    numberEpochs = 15
    plot = True
    eigen_compute = True
    oscill_compute = True
    div_compute = True
    dump_compute = True
    save_imgs = False
    comment = ''


    ########################
    for gammaFw, betaFB, alphaRec in  UsedForLearningHyper:

        model= PCMLP(0.33,alphaRec,betaFB,gammaFw)
        checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_G{gammaFw}_B{betaFB}_A{alphaRec}.pth"))
        model.load_state_dict(checkpointPhase["module"])
        for name, p in model.named_parameters():
            tmp = p.detach().numpy()
            if name=='fcAB.weight':
                W12 = tmp
            if name=='fcBA.weight':
                W21 = tmp
            if name=='fciA.weight':
                W01 = tmp
            if name=='fcAi.weight':
                W10 = tmp
            if name=='fcBC.weight':
                W23 = tmp
            if name=='fcCB.weight':
                W32 = tmp
            

        
        d = W01.shape[1]
        Z = np.zeros((d,d))

        for alpha in alphaR :
            good_params = []
            over_one = []
            under_one = []

            if eigen_compute :
                print('Computing eigenvalues...')
                for beta in betaR:
                    for i, gamma in enumerate(gammaR):
                        if beta + gamma > 1:
                            pass
                        else:

                            A11 = (1-beta-gamma) * np.eye(d) - alpha/d * W10.T.dot(W10)
                            A12 = beta * W21
                            A13 = np.zeros((d,d))
                            A21 = (1-beta-gamma) * gamma * W21.T - alpha/d * gamma * W12.dot(W10.T.dot(W10)) + alpha/d * W21.T
                            A22 = gamma * beta * W12.dot(W21) + (1-beta-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                            A23 = beta * W32
                            A31 = (1-beta-gamma) * gamma**2 * W23.dot(W12) - alpha/d * gamma**2 *W23.dot(W12.dot(W10.T.dot(W10))) + alpha/d * gamma * W23.dot(W32.T)
                            A32 = beta * gamma **2 * W12.dot(W21) + (1-beta-gamma) * gamma * W23 - alpha/d * gamma * W23.dot(W21.T.dot(W21)) + alpha/d * W32.T
                            A33 = beta * gamma * W23 * W32 + (1-gamma) * np.eye(d) - alpha/d * W32.T.dot(W32)
                            A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])

                            rho,_ = eigs(A,k=1,which='LM',tol=10**-3) # ie spectral radius
                            RhoCloseToOne(rho,good_params,over_one,under_one,beta,gamma,tol1 = tolOne,tolover=tolOver,tolunder=tolUnder)


                np.save(os.path.join('parameters_setup',f'{comment}good_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'),good_params)
                np.save(os.path.join('parameters_setup',f'{comment}over_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'),over_one)
                np.save(os.path.join('parameters_setup',f'{comment}under_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'),under_one)
                
            good_params = np.load(os.path.join('parameters_setup',f'{comment}good_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'))
            over_one = np.load(os.path.join('parameters_setup',f'{comment}over_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'))
            under_one =  np.load(os.path.join('parameters_setup',f'{comment}under_params_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.npy'))
            
            if plot:
                print('Saving plots...')
                plt.figure()
                if not len(over_one) ==0 :
                    plt.scatter(*zip(*over_one),color='blue',label=f'$\\rho$ > {1+tolOver}',s=0.1)
                if not len(under_one) ==0 :
                    plt.scatter(*zip(*under_one),color='green',label=f'$\\rho$ < {1-tolUnder}',s=0.1)
                if not len(good_params) ==0 :
                    plt.scatter(*zip(*good_params),color='red',label=f'$\\rho$ = 1 +-{tolOne}',s=0.5)
                plt.scatter(betaFB,gammaFw, label='Used during learning')
                x = np.linspace(0,1,100)
                plt.plot(x,1-x,linestyle='dashed',label='$\lambda+\\beta = 1$',color='red')
                plt.xlabel('$\lambda$')
                plt.xlim((0,1))
                plt.ylim((0,1))
                plt.ylabel('$\\beta$')
                plt.title(f'Potential oscillations for $\\alpha = {alpha}$')
                plt.legend()
                plt.savefig(os.path.join('parameters_plot',f'{comment}potential_good_parameters_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.png'))
                plt.close()

            osci_eigv = []
            div_eigv = []
            conv_eigv = []
            if oscill_compute :
                print('Computing PGO...')

                for beta,gamma in good_params:
                    

                    A11 = (1-beta-gamma) * np.eye(d) - alpha/d * W10.T.dot(W10)
                    A12 = beta * W21
                    A13 = np.zeros((d,d))
                    A21 = (1-beta-gamma) * gamma * W21.T - alpha/d * gamma * W12.dot(W10.T.dot(W10)) + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-beta-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A23 = beta * W32
                    A31 = (1-beta-gamma) * gamma**2 * W23.dot(W12) - alpha/d * gamma**2 *W23.dot(W12.dot(W10.T.dot(W10))) + alpha/d * gamma * W23.dot(W32.T)
                    A32 = beta * gamma **2 * W12.dot(W21) + (1-beta-gamma) * gamma * W23 - alpha/d * gamma * W23.dot(W21.T.dot(W21)) + alpha/d * W32.T
                    A33 = beta * gamma * W23 * W32 + (1-gamma) * np.eye(d) - alpha/d * W32.T.dot(W32)
                    A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if np.isreal(eig) or abs(1-abs(eig)) >= tolOne :
                            pass
                        else:
                            #print(eig)
                            osci_eigv.append((beta,gamma,v[i]))

                osci_imgs = [(beta,gamma,np.real(y)) for beta,gamma,y in osci_eigv]#[(beta,gamma,np.real(y[:196])) for beta,gamma,y in osci_eigv]

                unflattened_imgs =dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(osci_imgs)]) #dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_good_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if div_compute:
                print('Computing PGE...')
                # if len(over_one)>10:
                #     over_one = over_one[:10]

                for beta,gamma in over_one:
                    
                    A11 = (1-beta-gamma) * np.eye(d) - alpha/d * W10.T.dot(W10)
                    A12 = beta * W21
                    A13 = np.zeros((d,d))
                    A21 = (1-beta-gamma) * gamma * W21.T - alpha/d * gamma * W12.dot(W10.T.dot(W10)) + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-beta-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A23 = beta * W32
                    A31 = (1-beta-gamma) * gamma**2 * W23.dot(W12) - alpha/d * gamma**2 *W23.dot(W12.dot(W10.T.dot(W10))) + alpha/d * gamma * W23.dot(W32.T)
                    A32 = beta * gamma **2 * W12.dot(W21) + (1-beta-gamma) * gamma * W23 - alpha/d * gamma * W23.dot(W21.T.dot(W21)) + alpha/d * W32.T
                    A33 = beta * gamma * W23 * W32 + (1-gamma) * np.eye(d) - alpha/d * W32.T.dot(W32)
                    A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])

                    w, v = eigs(A,k=1,which='LM',tol=10**-3) 
                    
                    div_eigv.append((beta,gamma,v))

                div_imgs = [(beta,gamma,np.real(y)) for beta,gamma,y in div_eigv]
                print(len(div_imgs))

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(div_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_over_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if dump_compute:
                print('Computing PGC...')
                if len(under_one)>10:
                    under_one = under_one[:10]

                for beta,gamma in under_one:
                    
                    A11 = (1-beta-gamma) * np.eye(d) - alpha/d * W10.T.dot(W10)
                    A12 = beta * W21
                    A13 = np.zeros((d,d))
                    A21 = (1-beta-gamma) * gamma * W21.T - alpha/d * gamma * W12.dot(W10.T.dot(W10)) + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-beta-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A23 = beta * W32
                    A31 = (1-beta-gamma) * gamma**2 * W23.dot(W12) - alpha/d * gamma**2 *W23.dot(W12.dot(W10.T.dot(W10))) + alpha/d * gamma * W23.dot(W32.T)
                    A32 = beta * gamma **2 * W12.dot(W21) + (1-beta-gamma) * gamma * W23 - alpha/d * gamma * W23.dot(W21.T.dot(W21)) + alpha/d * W32.T
                    A33 = beta * gamma * W23 * W32 + (1-gamma) * np.eye(d) - alpha/d * W32.T.dot(W32)
                    A = np.block([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
                    w, v = eigs(A,k=1,which='LM',tol=10**-3) 
                    
                    conv_eigv.append((beta,gamma,v))

                conv_imgs = [(beta,gamma,np.real(y)) for beta,gamma,y in conv_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(conv_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_under_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            
if __name__ == "__main__":
    main()