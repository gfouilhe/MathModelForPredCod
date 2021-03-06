import numpy as np
import matplotlib.pyplot as plt
from model import PCMLP
import torch
import os

def main():

    g = []
    b = []
    alphaR = [0.1]#0.01,0.05,0.1,0.5,1,10,  100]
    numberEpochs = 15

    # gammaR = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # betaR = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    trained_params_couples = [(0.1,0.8),(0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]
    gammaR = list(np.linspace(0,1,10))
    betaR = list(np.linspace(0,1,10))

    for gammaFw, betaFB in trained_params_couples:

        for alpha in alphaR:

            print(f'trained in gamma = {gammaFw} and beta = {betaFB} and alpha = {alpha}')
            g.append(gammaFw)
            b.append(betaFB)
            model= PCMLP(0.33,alpha,betaFB,gammaFw)
            checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_G{gammaFw}_B{betaFB}_A{alpha}.pth"))
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

            good_params = []
            over_one = []
            under_one = []

            if betaFB + gammaFw > 1:
                pass
            else:
                ax = plt.gca()
                fig = plt.gcf()
                ax.cla()
                circle = plt.Circle((0,0),1,color='r',fill=False)
                ax.add_patch(circle)

                for gamma in gammaR:
                    for beta in betaR:

                        if gamma+beta>1:

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
                            w, v = np.linalg.eig(A)
                            
                            for eig in w:
                                re = np.real(eig)
                                im = np.imag(eig)
                                ax.plot(re,im,'o',color='blue')

                plt.title(f'eigs_G{gammaFw}_B{betaFB}_A{alpha}')
                plt.savefig(os.path.join('eigenvalues_plot',f'eigs_G{gammaFw}_B{betaFB}_A{alpha}.png'))
                plt.close()


    
    plt.scatter(g,b)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Used parameters')
    plt.savefig(os.path.join('eigenvalues_plot',f'used_parameters.png'))

if __name__ == "__main__":
    main()