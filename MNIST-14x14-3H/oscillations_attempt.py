import torch
# import torchvision.transforms as transforms
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle

# good_parameters = np.load(os.path.join('oscillations_parameters_setup','good_params.npy'))

# params_and_imgs = np.load(os.path.join('oscillations_parameters_setup','paramsandimgs.npy'))

with open(os.path.join('oscillations_parameters_setup','params_dictionary.pkl'), 'rb') as f:
    params_and_imgs = pickle.load(f)

params_list = [param for _,param in params_and_imgs.items()]

rgba2gray = lambda x: np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])

# imgs = [rgba2gray(255 *plt.imread(os.path.join('oscillations_parameters_setup',f'img{i}.png'))).reshape((28,28)) for i in range(1,40)]

batchSize = 1


timeSteps = 50

actA = np.zeros((len(params_list),timeSteps+1,196))
actB = np.zeros((len(params_list),timeSteps+1,196))
actC = np.zeros((len(params_list),timeSteps+1,196))
actO = np.zeros((len(params_list),timeSteps+1,10))


for i,im in enumerate(params_list):
    beta,gamma,img = im
    pcmodel = PCMLP(0.33,0.01,betaFB=beta,gammaFw=gamma)
    checkpointPhase = torch.load(os.path.join('models',"PC_E19_I0.pth"))
    pcmodel.load_state_dict(checkpointPhase["module"])
    img = torch.from_numpy(img.astype('float32'))
    aTemp = torch.zeros(batchSize, 196)
    bTemp = torch.zeros(batchSize, 196)
    cTemp = torch.zeros(batchSize, 196)
    oTemp = torch.zeros(batchSize, 10)
    _, _ , aTemp, bTemp, cTemp, oTemp, _, _ = pcmodel(img.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'forward')
    actA[i,0,:] = aTemp.detach().numpy()
    actB[i,0,:] = bTemp.detach().numpy()
    actC[i,0,:] = cTemp.detach().numpy()
    actO[i,0,:] = oTemp.detach().numpy()
    for t in range(timeSteps):
        _, _, aTemp, bTemp, cTemp, oTemp, _, _ =  pcmodel(img.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'full')
        actA[i,t+1,:] = aTemp.detach().numpy()
        actB[i,t+1,:] = bTemp.detach().numpy()
        actC[i,t+1,:] = cTemp.detach().numpy()
        actO[i,t+1,:] = oTemp.detach().numpy()

actA = np.linalg.norm(actA,axis=2)
actB = np.linalg.norm(actB,axis=2)
actC = np.linalg.norm(actC,axis=2)
actO = np.linalg.norm(actO,axis=2)

for i, _ in enumerate(params_list):
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.plot(actA[i])
    plt.title('Activations norm of layer A')
    plt.subplot(1,4,2)
    plt.plot(actB[i])
    plt.title('Activations norm of layer B')
    plt.subplot(1,4,3)
    plt.plot(actC[i])
    plt.title('Activations norm of layer C')
    plt.subplot(1,4,4)
    plt.plot(actO[i])
    plt.title('Activations norm of layer O')
    plt.savefig(os.path.join('oscillations_attempt_plot_norm',f'img{i}'))
    plt.close()



