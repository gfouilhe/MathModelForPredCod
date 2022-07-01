
import torch
from network import UniDimModel
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle 
from scipy.sparse.linalg import eigs
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def RhoCloseToOne(rho,l,over,under,lamb,beta,tol1 = 10**-2,tolover = 0.5, tolunder = 0.2):
    rho = abs(rho)
    if abs(1-rho) < tol1:
        l.append((lamb,beta))
    elif rho > 1 + tolover :
        over.append((lamb,beta))
    elif rho < 1 - tolunder :
        under.append((lamb,beta))

def main():
    
    #-------Parameters-----

    # These are the parameters used for computing eigenvalues, not those used to learn the weights.
    alphaR = [0.01,0.05,0.1,0.25]
    lambR = list(np.arange(0,1,0.005))[1:] 
    betaR = list(np.arange(0,1,0.005))[1:]
    #

    # Parameters used for learning :
    UsedForLearningHyper = [(0.6,0.2,0.01),(0.33,0.33,0.01),(0.5,0.25,0.01),(0.7,0.1,0.01),(0.85,0.05,0.01),(0.95,0.01,0.01)]
    #

    # Tolerences :
    tolOne = 0.001
    tolOver = 0.01
    tolUnder = 0.01

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
    for betaFw, lambdaBw, alphaRec in  UsedForLearningHyper:

        model = UniDimModel(0.33,alphaRec,betaFw,lambdaBw)
        checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth"))
        model.load_state_dict(checkpointPhase["module"])
        for name, p in model.named_parameters():
            tmp = p.detach().numpy()
            if name=='fc12.weight':
                W12 = tmp[0][0]
            if name=='fc21.weight':
                W21 = tmp[0][0]
            if name=='fc01.weight':
                W01 = tmp[0][0]
            if name=='fc10.weight':
                W10 = tmp[0][0]
            if name=='fc23.weight':
                W23 = tmp[0][0]
            if name=='fc32.weight':
                W32 = tmp[0][0]

        
        

        for alpha in alphaR :
            good_params = []
            over_one = []
            under_one = []

            if eigen_compute :
                print('Computing eigenvalues...')
                for beta in betaR:
                    for i, lamb in enumerate(lambR):
                        if beta + lamb > 1:
                            pass
                        else:
                            
                            Wf = np.array([[0, 0, 0],
                                           [W12, 0, 0],
                                           [0, W23, 0]])
                            Wb = np.array([[0, W21, 0],
                                           [0, 0, W32],
                                           [0, 0, 0]])
                            D = np.array([[(1-beta-lamb), 0, 0],
                                           [0, (1-beta-lamb), 0],
                                           [0, 0, (1-beta)]])
                            E = np.array([[-W10**2, 0, 0],
                                           [W21, -W21**2, 0],
                                           [0, W32, -W32**2]])
                            
                            Inv = np.linalg.inv(np.eye(3)- beta*Wf.astype(float))
                            
                            A = Inv.dot((lamb*Wb + D + alpha*E))
                            A = A.astype(float)
                            rho,_ = eigs(A,k=1,which='LM',tol=10**-3) # ie spectral radius
                            RhoCloseToOne(rho,good_params,over_one,under_one,lamb,beta,tol1 = tolOne,tolover=tolOver,tolunder=tolUnder)


                np.save(os.path.join('parameters_setup',f'{comment}good_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'),good_params)
                np.save(os.path.join('parameters_setup',f'{comment}over_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'),over_one)
                np.save(os.path.join('parameters_setup',f'{comment}under_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'),under_one)
                
            good_params = np.load(os.path.join('parameters_setup',f'{comment}good_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'))
            over_one = np.load(os.path.join('parameters_setup',f'{comment}over_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'))
            under_one =  np.load(os.path.join('parameters_setup',f'{comment}under_params_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.npy'))
            
            if plot:
                print('Saving plots...')
                plt.figure()
                if not len(over_one) == 0 :
                    plt.scatter(*zip(*over_one),color='blue',label=f'$\\rho$ > {1+tolOver}',s=0.1)
                if not len(under_one) == 0:
                    plt.scatter(*zip(*under_one),color='green',label=f'$\\rho$ < {1-tolUnder}',s=0.1)
                if not len(good_params) == 0:
                    plt.scatter(*zip(*good_params),color='red',label=f'$\\rho$ = 1 +/- {tolOne}',s=0.5)
                plt.scatter(lambdaBw,betaFw, label='Used during learning')
                x = np.linspace(0,1,100)
                plt.plot(x,1-x,linestyle='dashed',label='$\lambda+\\beta = 1$',color='red')
                plt.xlabel('$\lambda$')
                plt.xlim((0,1))
                plt.ylim((0,1))
                plt.ylabel('$\\beta$')
                plt.title(f'Potential oscillations for $\\alpha = {alpha}$')
                plt.legend()
                plt.savefig(os.path.join('parameters_plot',f'{comment}potential_good_parameters_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.png'))
                plt.close()

            osci_eigv = []
            div_eigv = []
            conv_eigv = []
            if oscill_compute :
                print('Computing PGO...')

                for lamb,beta in good_params:

                    Wf = np.array([[0, 0, 0],
                                   [W12, 0, 0],
                                   [0, W23, 0]])
                    Wb = np.array([[0, W21, 0],
                                   [0, 0, W32],
                                   [0, 0, 0]])
                    D = np.array([[(1-beta-lamb), 0, 0],
                                  [0, (1-beta-lamb), 0],
                                  [0, 0, (1-beta)]])
                    E = np.array([[-W10**2, 0, 0],
                                  [W21, -W21**2, 0],
                                  [0, W32, -W32**2]])

                    Inv = np.linalg.inv(np.eye(3)- beta*Wf)

                    A = Inv.dot((lamb*Wb + D + alpha*E))

                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if np.isreal(eig) or abs(1-abs(eig)) >= tolOne :
                            pass
                        else:
                            #print(eig)
                            osci_eigv.append((lamb,beta,v[i]))
                            print(eig, abs(eig))

                osci_imgs = [(lamb,beta,np.real(y)) for lamb,beta,y in osci_eigv]#[(beta,gamma,np.real(y[:196])) for beta,gamma,y in osci_eigv]

                unflattened_imgs =dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(osci_imgs)]) #dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(osci_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_good_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if div_compute:
                print('Computing PGE...')
                if len(over_one)>10:
                    over_one = over_one[:10]

                for lamb,beta in over_one:
                    
                    Wf = np.array([[0, 0, 0],
                                   [W12, 0, 0],
                                   [0, W23, 0]])
                    Wb = np.array([[0, W21, 0],
                                   [0, 0, W32],
                                   [0, 0, 0]])
                    D = np.array([[(1-beta-lamb), 0, 0],
                                  [0, (1-beta-lamb), 0],
                                  [0, 0, (1-beta)]])
                    E = np.array([[-W10**2, 0, 0],
                                  [W21, -W21**2, 0],
                                  [0, W32, -W32**2]])

                    Inv = np.linalg.inv(np.eye(3)- beta*Wf)

                    A = Inv.dot((lamb*Wb + D + alpha*E))
                    w, v = np.linalg.eig(A)

                    for i,eig in enumerate(w):
                        if abs(eig)> 1 + tolOver:
                            div_eigv.append((lamb,beta,v[i]))

                div_imgs = [(lamb,beta,np.real(y)) for lamb,beta,y in div_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(div_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_over_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            if dump_compute:
                print('Computing PGC...')
                if len(under_one)>10:
                    under_one = under_one[:10]

                for lamb,beta in under_one:
                    
                    Wf = np.array([[0, 0, 0],
                                   [W12, 0, 0],
                                   [0, W23, 0]])
                    Wb = np.array([[0, W21, 0],
                                   [0, 0, W32],
                                   [0, 0, 0]])
                    D = np.array([[(1-beta-lamb), 0, 0],
                                  [0, (1-beta-lamb), 0],
                                  [0, 0, (1-beta)]])
                    E = np.array([[-W10**2, 0, 0],
                                  [W21, -W21**2, 0],
                                  [0, W32, -W32**2]])

                    Inv = np.linalg.inv(np.eye(3)- beta*Wf)

                    A = Inv.dot((lamb*Wb + D + alpha*E))

                    w, v = np.linalg.eig(A)
                    for i,eig in enumerate(w):
                        if abs(eig) < 1 - tolUnder:
                            conv_eigv.append((lamb,beta,v[i]))

                conv_imgs = [(lamb,beta,np.real(y)) for lamb,beta,y in conv_eigv]

                unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(conv_imgs)]) #img[0:1] are parameters beta and gamma
                with open(os.path.join('parameters_setup',f'{comment}_under_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl'), 'wb') as f:
                    pickle.dump(unflattened_imgs, f)
            
            if save_imgs:
                print('Saving imgs...')
                with open(os.path.join('parameters_setup',f'{comment}_good_params_dictionary_G{betaFw}_B{lambdaBw}_A{alphaRec}_{alpha}.pkl'), 'rb') as f:
                    unflattened_imgs = pickle.load(f)
                unflattened_imgs_list = [img for _,img in unflattened_imgs.items()]
                print('Number of PGO imgs : ', len(unflattened_imgs_list))
                for i,img in enumerate(unflattened_imgs_list):
                    if i<10:
                        _,_,img = img
                        print(img.shape)
                        plt.imsave(os.path.join('PGOImgs',f'{comment}img_G{betaFw}_B{lambdaBw}_A{alphaRec}_{alpha}.png'),img, cmap='gray')

if __name__ == "__main__":
    main()