import numpy as np
import matplotlib.pyplot as plt
from model import PCMLP
import torch
import os

def main():

    gamma = 0.33
    beta = 0.33
    alpha = 120
    numberEpochs = 10

    trained_params_couples = [(0.33,0.33,100)]#[(0.1,0.8), (0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]

    for gammaFw, betaFB, alphaRec in trained_params_couples:

    

        print(f'trained in gamma = {gammaFw} and beta = {betaFB} and alpha = {alphaRec}')

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

        
        d = W01.shape[1]

        ax = plt.gca()
        fig = plt.gcf()
        ax.cla()
        circle = plt.Circle((0,0),1,color='r',fill=False)
        ax.add_patch(circle)

    
        A11 = (1-beta-gamma) * np.eye(d) - alpha/d * W10.T.dot(W10)
        A12 = beta * W21
        A21 = (1-beta-gamma) * gamma * W12 - alpha/d * gamma * W12.dot(W10.T.dot(W10)) + alpha/d * W21.T
        A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)

        A = np.block([[A11,A12],[A21,A22]])
        w, v = np.linalg.eig(A)

        for eig in w:
            re = np.real(eig)
            im = np.imag(eig)
            ax.plot(re,im,'o',color='blue')
        plt.title(f'eigs_G{gammaFw}_B{betaFB}_A{alphaRec}_G{gamma}_B{beta}_A{alpha}')
        plt.savefig(os.path.join('eigenvalues_plot',f'eigs_G{gammaFw}_B{betaFB}_A{alphaRec}_G{gamma}_B{beta}_A{alpha}.png'))
        plt.close()

if __name__ == "__main__":
    main()