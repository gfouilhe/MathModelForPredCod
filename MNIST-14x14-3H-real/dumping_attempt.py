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

timeSteps = 150
normA = []
normB = []
normO = []
beta = 0.05
gamma = 0.8
alpha = 0.01
eigv = []
model= PCMLP(0.33,alpha,beta,gamma).to(device)
    
checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0_G{0.7}_B{0.1}_A{0.01}.pth"))
print(gamma,beta,alpha)
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    tmp = p.detach().cpu().numpy()
    if name=='fcAB.weight':
        W12 = tmp
    if name=='fcBA.weight':
        W21 = tmp
    if name=='fciA.weight':
        W01 = tmp
    if name=='fcAi.weight':
        W10 = tmp

d = W12.shape[1]
A11 = (1-beta-gamma) * np.eye(d)
A12 = beta * W21
A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
A = np.block([[A11,A12],[A21,A22]])

_, v = np.linalg.eig(A)
for i,eig in enumerate(v):
        eigv.append((beta,gamma,eig))

imgs = [(betab,gammab,np.real(y[:196])) for betab,gammab,y in eigv]

unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2].reshape((14,14)))) for i, img in enumerate(imgs)])

params_list = [param for _,param in unflattened_imgs.items()]
actA = np.zeros((len(params_list),timeSteps+1,196))
actB = np.zeros((len(params_list),timeSteps+1,196))
actO = np.zeros((len(params_list),timeSteps+1,10))
for i,im in enumerate(params_list):
    beta,gamma,img = im
    pcmodel = PCMLP(0.33,alpha,betaFB=beta,gammaFw=gamma).to(device)
    checkpointPhase = torch.load(os.path.join('models',f"PC_E19_I0_G{0.7}_B{0.1}_A{0.01}.pth"))
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



for i, _ in enumerate(params_list):

    nA = normA[0]
    nB = normB[0]
    nO = normO[0]

    if np.size(nA)==0:
        pass
    else:
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(nA[i])
        plt.subplot(1,3,2)
        plt.plot(nB[i])
        plt.subplot(1,3,3)
        plt.plot(nO[i])
        plt.savefig(os.path.join('dump_attempt_plot_norm',f'actplot_{alpha}_{i}.png'))
        plt.close()



