
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

def main():
    
    #-------Parameters-----

    # These are the parameters used for computing eigenvalues, not those used to learn the weights.
    alphaR = [0.01,0.05,0.1,0.25]
    betaR = list(np.arange(0,1,0.005))[1:] 
    gammaR = list(np.arange(0,1,0.005))[1:]
    #

    # Parameters used for learning :
    gammaFwL = [0.7]
    alphaRecL = [0.01]
    betaFBL = [0.1]
    UsedForLearningHyper = [(0.7,0.1,0.01),(0.33,0.33,0.01),(0.85,0.05,0.01),(0.95,0.01,0.01)]
    #

    # Tolerences :
    tolOne = 0.01
    tolOver = 0.5
    tolUnder = 0.2

    # Others :
    numberEpochs = 20
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

            # # The following are for CVMLP

            # if name=='fciA.fc_r.weight':
            #     W01R = tmp
            # if name=='fciA.fc_i.weight':
            #     W01I = tmp
            # if name=='fcAB.fc_r.weight':
            #     W12R = tmp
            # if name=='fcAB.fc_i.weight':
            #     W12I = tmp
            # if name=='fcAi.fc_r.weight':
            #     W10R = tmp
            # if name=='fcAi.fc_i.weight':
            #     W10I = tmp
            # if name=='fcBA.fc_r.weight':
            #     W21R = tmp
            # if name=='fcBA.fc_i.weight':
            #     W21I = tmp

        # # For CVMLP :
        # d = W01R.shape[1]
        # Z = np.zeros((d,d))
        # W01 = np.block([[W01R,Z],[Z,W01I]])
        # W10 = np.block([[W10R,Z],[Z,W10I]])
        # W12 = np.block([[W12R,Z],[Z,W12I]])
        # W21 = np.block([[W21R,Z],[Z,W21I]])=

        
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

                            A11 = (1-beta-gamma) * np.eye(d)
                            A12 = beta * W21
                            A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
                            A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)

                            A = np.block([[A11,A12],[A21,A22]])

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
                plt.scatter(*zip(*over_one),color='blue',label=f'$\\rho$ > {1+tolOver}',s=0.1)
                plt.scatter(*zip(*under_one),color='green',label=f'$\\rho$< {1-tolOver}',s=0.1)
                plt.scatter(*zip(*good_params),color='red',label='$\\rho$ = 1',s=0.5)
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
                    

                    A11 = (1-beta-gamma) * np.eye(d)
                    A12 = beta * W21
                    A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A = np.block([[A11,A12],[A21,A22]])

                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if np.isreal(eig) or abs(1-abs(eig)) >= tolOne :
                            pass
                        else:
                            #print(eig)
                            osci_eigv.append((beta,gamma,v[i]))

                osci_imgs = [(beta,gamma,np.real(y[:196])) for beta,gamma,y in osci_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_good_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if div_compute:
                print('Computing PGE...')
                if len(over_one)>10:
                    over_one = over_one[:10]

                for beta,gamma in over_one:
                    
                    A11 = (1-beta-gamma) * np.eye(d)
                    A12 = beta * W21
                    A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A = np.block([[A11,A12],[A21,A22]])

                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if np.isreal(eig) or (abs(eig) - 1 > tolOver and abs(eig)> 1) :
                            pass
                        else:
                            #print(eig)
                            div_eigv.append((beta,gamma,v[i]))

                div_imgs = [(beta,gamma,np.real(y[:196])) for beta,gamma,y in div_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(div_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_over_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if dump_compute:
                print('Computing PGC...')
                if len(under_one)>10:
                    under_one = under_one[:10]

                for beta,gamma in under_one:
                    
                    A11 = (1-beta-gamma) * np.eye(d)
                    A12 = beta * W21
                    A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
                    A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
                    A = np.block([[A11,A12],[A21,A22]])

                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if np.isreal(eig) or (1 - abs(eig) > tolOver and abs(eig)<1):
                            pass
                        else:
                            #print(eig)
                            conv_eigv.append((beta,gamma,v[i]))

                conv_imgs = [(beta,gamma,np.real(y[:196])) for beta,gamma,y in conv_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(conv_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_under_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            
            if save_imgs:
                print('Saving imgs...')
                with open(os.path.join('parameters_setup',f'{comment}params_dictionary__G{gammaFw}_B{betaFB}_A{alphaRec}__G{gamma}_B{beta}_A{alpha}.pkl'), 'rb') as f:
                    unflattened_imgs = pickle.load(f)
                unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
                print('Number of PGO imgs : ', len(unflattened_imgs_list))
                for i,img in enumerate(unflattened_imgs_list):
                    _,_,img = img
                    plt.imsave(os.path.join('PGOImgs',f'{comment}img_G{gammaFw}_B{betaFB}_A{alphaRec}__G{gamma}_B{beta}_A{alpha}_{i}.png'),img, cmap='gray')

if __name__ == "__main__":
    main()