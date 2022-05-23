import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

batchSize = 1

timeSteps = 50
normA = []
normB = []
normO = []
gamma_beta_couples = [(0.1,0.7),(0.25,0.5),(0.33,0.33),(0.5,0.25),(0.7,0.1)]
alpha_range = [0.01,0.05,0.1,0.5]
for gamma, beta in gamma_beta_couples:
    normA = []
    normB = []
    normO = []
    for alpha in alpha_range:
        with open(os.path.join('oscillations_parameters_setup',f'params_dictionary_G{gamma}_B{beta}_A{alpha}.pkl'), 'rb') as f:
            params_and_imgs = pickle.load(f)

        params_list = [param for _,param in params_and_imgs.items()]
        print(len(params_list))
        actA = np.zeros((len(params_list),timeSteps+1,196))
        actB = np.zeros((len(params_list),timeSteps+1,196))
        actO = np.zeros((len(params_list),timeSteps+1,10))
        for i,im in enumerate(params_list):
            betab,gammab,img = im
            pcmodel = PCMLP(0.33,alpha,betaFB=betab,gammaFw=gammab).to(device)
            checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0_G{gamma}_B{beta}_A{alpha}.pth"))
            pcmodel.load_state_dict(checkpointPhase["module"])
            img = torch.from_numpy(img.astype('float32')).to(device)
            aTemp = torch.zeros(batchSize, 196).cuda()
            bTemp = torch.zeros(batchSize, 196).cuda()
            oTemp = torch.zeros(batchSize, 10).cuda()
            _, _ , aTemp, bTemp, oTemp, _ = pcmodel(img.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
            actA[i,0,:] = aTemp.detach().cpu().numpy()
            actB[i,0,:] = bTemp.detach().cpu().numpy()
            actO[i,0,:] = oTemp.detach().cpu().numpy()
            for t in range(timeSteps):
                _, _, aTemp, bTemp, oTemp, _ =  pcmodel(img.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')
                actA[i,t+1,:] = aTemp.detach().cpu().numpy()
                actB[i,t+1,:] = bTemp.detach().cpu().numpy()
                actO[i,t+1,:] = oTemp.detach().cpu().numpy()
        normA.append(np.linalg.norm(actA,axis=2))
        normB.append(np.linalg.norm(actB,axis=2))
        normO.append(np.linalg.norm(actO,axis=2))

    # actA = np.linalg.norm(actA,axis=2)
    # actB = np.linalg.norm(actB,axis=2)
    # actO = np.linalg.norm(actO,axis=2)


    for j,alpha in enumerate(alpha_range):
        with open(os.path.join('oscillations_parameters_setup',f'params_dictionary_G{gamma}_B{beta}_A{alpha}.pkl'), 'rb') as f:
            params_and_imgs = pickle.load(f)
        params_list = [param for _,param in params_and_imgs.items()]
        for i, _ in enumerate(params_list):

            nA = normA[j]
            print(nA)
            nB = normB[j]
            nO = normO[j]

            if np.size(normA[j])==0:
                print('yoaozeazeazea')
                pass
            else:
                print('hi')
                plt.figure(figsize=(15,5))
                plt.subplot(1,3,1)
                plt.plot(nA[i])
                plt.subplot(1,3,2)
                plt.plot(nB[i])
                plt.subplot(1,3,3)
                plt.plot(nO[i])
                plt.savefig(os.path.join('oscillations_attempt_plot_norm',f'actplot_G{gamma}_B{beta}_A{alpha}_{i}.png'))
                plt.close()



