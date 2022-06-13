import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle

batchSize = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():

    #------ Parameteters -------

    mode = 'oscillations' # 'explode' , 'dump', 'oscillations'


    UsedForLearningHyper =  [(0.7,0.1,0.01),(0.33,0.33,0.01),(0.85,0.05,0.01),(0.95,0.01,0.01)]
    comment = ''
    alphaR = [0.01,0.05,0.1,0.25]
    numberEpochs = 20
    timeSteps = 50
    commentact = 'linear' #'tanh' ; 'relu'
    
    if commentact == 'linear' :
        activation = lambda x: x 
    elif commentact == 'tanh':
        activation = torch.tanh
    elif commentact == 'relu':
        activation = torch.relu

    for gammaFw, betaFB, alphaRec in UsedForLearningHyper:

        for alpha in alphaR:


            if mode=='oscillations':

                path = os.path.join('parameters_setup',f'{comment}_good_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl')

            if mode=='explode':

                path = os.path.join('parameters_setup',f'{comment}_over_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl')
                
            if mode=='dump':
                
                path = os.path.join('parameters_setup',f'{comment}_under_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}_{alpha}.pkl')

            with open(path,'rb') as f:
                    params_and_imgs = pickle.load(f)

            params_list = [param for _,param in params_and_imgs.items()]
            actA = np.zeros((len(params_list),timeSteps+1,196))
            actB = np.zeros((len(params_list),timeSteps+1,196))
            actO = np.zeros((len(params_list),timeSteps+1,10))
            for i,im in enumerate(params_list):
                beta,gamma,img = im
                aTemp = img[:196]
                aTemp = torch.from_numpy(aTemp.astype('float32')).to(device).view(batchSize,-1)
                iTemp = torch.clone(aTemp)
                bTemp = img[196:]
                bTemp = torch.from_numpy(bTemp.astype('float32')).to(device).view(batchSize,-1)
                oTemp = torch.zeros(batchSize, 10)
                pcmodel = PCMLP(0.33,alphaRec=alpha,betaFB=beta,gammaFw=gamma,activation_function=activation).to(device)
                checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_G{gammaFw}_B{betaFB}_A{alphaRec}.pth"))
                pcmodel.load_state_dict(checkpointPhase["module"])

                iTemp.requires_grad = True
                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                _, iTemp,_, _, _, _ = pcmodel(iTemp, aTemp, bTemp, oTemp, 'reconstruction')

                actA[i,0,:] = aTemp.detach().cpu().numpy()
                actB[i,0,:] = bTemp.detach().cpu().numpy()
                actO[i,0,:] = oTemp.detach().cpu().numpy()

                for t in range(timeSteps):
                    
                    _, _, aTemp, bTemp, oTemp, _ =  pcmodel(iTemp.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')
                    actA[i,t+1,:] = aTemp.detach().cpu().numpy()
                    actB[i,t+1,:] = bTemp.detach().cpu().numpy()
                    actO[i,t+1,:] = oTemp.detach().cpu().numpy()

            normA = np.linalg.norm(actA,axis=2)
            normB = np.linalg.norm(actB,axis=2)
            normO = np.linalg.norm(actO,axis=2)

            for i, (gamma,beta,_) in enumerate(params_list):

                if np.size(normA[i])==0:
                    pass
                else:
                    plt.figure(figsize=(15,5))
                    plt.subplot(1,3,1)
                    plt.plot(normA[i])
                    plt.title('Layer A')
                    plt.subplot(1,3,2)
                    plt.plot(normB[i])
                    plt.title('Layer B')
                    plt.subplot(1,3,3)
                    plt.plot(normO[i])
                    plt.title('Layer O')
                    plt.savefig(os.path.join(f'{mode}_attempt_plot_norm',f'{commentact}actplot_G{gammaFw}_B{betaFB}_A{alphaRec}_G{gamma}_B{beta}_A{alpha}_{i}.png'))
                    plt.close()

if __name__ == "__main__":
    main()