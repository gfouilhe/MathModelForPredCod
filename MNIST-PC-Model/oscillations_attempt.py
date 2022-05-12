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

good_parameters = np.load(os.path.join('oscillations_parameters_setup','good_params.npy'))


pcmodel = PCMLP(0.33,0.01,0.2,0.5)
checkpointPhase = torch.load(os.path.join('models',"PC_E19_I4.pth"))
pcmodel.load_state_dict(checkpointPhase["module"])

rgba2gray = lambda x: np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])

imgs = [rgba2gray(255 *plt.imread(os.path.join('oscillations_parameters_setup',f'img{i}.png'))).reshape((28,28)) for i in range(1,40)]

batchSize = 1


timeSteps = 50

actA = np.zeros((len(imgs),timeSteps+1,120))
actB = np.zeros((len(imgs),timeSteps+1,120))
actO = np.zeros((len(imgs),timeSteps+1,10))


for i,im in enumerate(imgs):
    img = torch.from_numpy(im.astype('float32'))
    aTemp = torch.zeros(batchSize, 120)
    bTemp = torch.zeros(batchSize, 120)
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

# actA = np.linalg.norm(actA,axis=2)
# actB = np.linalg.norm(actB,axis=2)
# actO = np.linalg.norm(actO,axis=2)

for i, _ in enumerate(imgs):
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(actA[i,:,0])
    plt.subplot(1,3,2)
    plt.plot(actB[i,:,0])
    plt.subplot(1,3,3)
    plt.plot(actO[i,:,0])
    plt.savefig(os.path.join('oscillations_attempt_plot_2',f'img{i}'))
    plt.close()



