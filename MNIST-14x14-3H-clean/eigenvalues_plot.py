import numpy as np
import matplotlib.pyplot as plt
from model import PCMLP
import torch
import os

def main():

    g = []
    b = []
    alphaR = [0]#0.01,0.05,0.1,0.5,1,10,  100]
    numberEpochs = 20

    # gammaR = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    # betaR = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    trained_params_couples = [(0.33,0.33),(0.5,0.25),(0.7,0.1),(0.85,0.05),(0.95,0.01)]

    for gamma, beta in trained_params_couples:

        for alpha in alphaR:
                
            g.append(gamma)
            b.append(beta)
            model= PCMLP(0.33,alpha,beta,gamma)
            checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_G{gamma}_B{beta}_A{alpha}.pth"))
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

            
            d = W01.shape[1]
            Z = np.zeros((d,d))

        
            good_params = []
            over_one = []
            under_one = []

        
            if beta + gamma > 1:
                pass
            else:

                A11 = (1-beta-gamma) * np.eye(d)
                A12 = beta * W21
                A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
                A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)

                A = np.block([[A11,A12],[A21,A22]])
                w, v = np.linalg.eig(A)
                ax = plt.gca()
                fig = plt.gcf()
                ax.cla()
                circle = plt.Circle((0,0),1,color='r',fill=False)
                ax.add_patch(circle)
                for eig in w:
                    print(eig,gamma,beta)
                    re = np.real(eig)
                    im = np.imag(eig)
                    ax.plot(re,im,'o',color='blue')
                    plt.title(f'eigs_G{gamma}_B{beta}_A{alpha}')
                plt.savefig(os.path.join('eigenvalues_plot',f'eigs_G{gamma}_B{beta}_A{alpha}.png'))
                plt.close()


    
    plt.scatter(g,b)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Used parameters')
    plt.savefig(os.path.join('eigenvalues_plot',f'used_parameters.png'))

if __name__ == "__main__":
    main()