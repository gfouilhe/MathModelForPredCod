import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle

batchSize = 1

timeSteps = 50
normA = []
normB = []
normO = []

for alpha in [0.01,0.02,0.05,0.1,0.25,0.5]:
    with open(os.path.join('oscillations_parameters_setup',f'params_dictionary_{alpha}.pkl'), 'rb') as f:
        params_and_imgs = pickle.load(f)

    params_list = [param for _,param in params_and_imgs.items()]
    actA = np.zeros((len(params_list),timeSteps+1,196))
    actB = np.zeros((len(params_list),timeSteps+1,196))
    actO = np.zeros((len(params_list),timeSteps+1,10))
    for i,im in enumerate(params_list):
        beta,gamma,img = im
        pcmodel = PCMLP(0.33,alpha,betaFB=beta,gammaFw=gamma)
        checkpointPhase = torch.load(os.path.join('models',"PC_E19_I4.pth"))
        pcmodel.load_state_dict(checkpointPhase["module"])
        img = torch.from_numpy(img.astype('float32'))
        aTemp = torch.zeros(batchSize, 196)
        bTemp = torch.zeros(batchSize, 196)
        oTemp = torch.zeros(batchSize, 10)
        _, _ , aTemp, bTemp, oTemp, _ = pcmodel(img.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
        actA[i,0,:] = aTemp.detach().numpy()
        actB[i,0,:] = bTemp.detach().numpy()
        actO[i,0,:] = oTemp.detach().numpy()
        for t in range(timeSteps):
            _, _, aTemp, bTemp, oTemp, _ =  pcmodel(img.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')
            actA[i,t+1,:] = aTemp.detach().numpy()
            actB[i,t+1,:] = bTemp.detach().numpy()
            actO[i,t+1,:] = oTemp.detach().numpy()
    normA.append(np.linalg.norm(actA,axis=2))
    normB.append(np.linalg.norm(actB,axis=2))
    normO.append(np.linalg.norm(actO,axis=2))

# actA = np.linalg.norm(actA,axis=2)
# actB = np.linalg.norm(actB,axis=2)
# actO = np.linalg.norm(actO,axis=2)

for i, _ in enumerate(params_list):
    for j,alpha in [0.01,0.02,0.05,0.1,0.25,0.5]:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(normA[j][i])
        plt.subplot(1,3,2)
        plt.plot(normB[j][i])
        plt.subplot(1,3,3)
        plt.plot(normO[j][i])
        plt.savefig(os.path.join('oscillations_attempt_plot_norm',f'img_{alpha}_{i}'))
        plt.close()



