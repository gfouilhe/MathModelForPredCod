import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle


with open(os.path.join('oscillations_parameters_setup','params_dictionary.pkl'), 'rb') as f:
    params_and_imgs = pickle.load(f)

params_list = [param for _,param in params_and_imgs.items()]

rgba2gray = lambda x: np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])


batchSize = 1


timeSteps = 50

actA = np.zeros((len(params_list),timeSteps+1,196))
actB = np.zeros((len(params_list),timeSteps+1,196))
actO = np.zeros((len(params_list),timeSteps+1,10))


for i,im in enumerate(params_list):
    beta,gamma,img = im
    pcmodel = PCMLP(0.33,0.01,betaFB=beta,gammaFw=gamma)
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

actA = np.linalg.norm(actA,axis=2)
actB = np.linalg.norm(actB,axis=2)
actO = np.linalg.norm(actO,axis=2)

for i, _ in enumerate(params_list):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(actA[i])
    plt.subplot(1,3,2)
    plt.plot(actB[i])
    plt.subplot(1,3,3)
    plt.plot(actO[i])
    plt.savefig(os.path.join('oscillations_attempt_plot_norm',f'img{i}'))
    plt.close()



