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
beta = 0.4
gamma = 0.4
alpha = 0.01
eigv = []
model= PCMLP(0.33,alpha,beta,gamma,complex_valued=True).to(device)
    
checkpointPhase = torch.load(os.path.join('models',f"PCC_E19_I0_G{0.6}_B{0.2}_A{0.01}.pth"))
print(gamma,beta,alpha)
model.load_state_dict(checkpointPhase["module"])
for name, p in model.named_parameters():
    tmp = p.detach().cpu().numpy()
    # if name=='fcAB.weight':
    #     W12 = tmp
    # if name=='fcBA.weight':
    #     W21 = tmp
    # if name=='fciA.weight':
    #     W01 = tmp
    # if name=='fcAi.weight':
    #     W10 = tmp
    if name=='fciA.fc_r.weight':
        W01R = tmp
    if name=='fciA.fc_i.weight':
        W01I = tmp
    if name=='fcAB.fc_r.weight':
        W12R = tmp
    if name=='fcAB.fc_i.weight':
        W12I = tmp
    if name=='fcAi.fc_r.weight':
        W10R = tmp
    if name=='fcAi.fc_i.weight':
        W10I = tmp
    if name=='fcBA.fc_r.weight':
        W21R = tmp
    if name=='fcBA.fc_i.weight':
        W21I = tmp
d = W01R.shape[1]
Z = np.zeros((d,d))
W01 = np.block([[W01R,Z],[Z,W01I]])
W10 = np.block([[W10R,Z],[Z,W10I]])
W12 = np.block([[W12R,Z],[Z,W12I]])
W21 = np.block([[W21R,Z],[Z,W21I]])
d = W12.shape[1]
A11 = (1-beta-gamma) * np.eye(d)
A12 = beta * W21
A21 = (1-beta-gamma) * gamma * W12 + alpha/d * W21.T
A22 = gamma * beta * W12.dot(W21) + (1-gamma) * np.eye(d) - alpha/d * W21.T.dot(W21)
A = np.block([[A11,A12],[A21,A22]])

w, v = np.linalg.eig(A)
for i,eig in enumerate(w):
    if abs(eig) > 1 + 10**-3:
        eigv.append((beta,gamma,np.real(v[i])))

imgs = [(betab,gammab,y[:196*2]) for betab,gammab,y in eigv]

unflattened_imgs = dict([(f"im{i})",(img[0],img[1],img[2])) for i, img in enumerate(imgs)])

params_list = [param for _,param in unflattened_imgs.items()]
actA = np.zeros((len(params_list),timeSteps+1,196))
actB = np.zeros((len(params_list),timeSteps+1,196))
actO = np.zeros((len(params_list),timeSteps+1,10))
for i,im in enumerate(params_list):
    beta,gamma,img = im
    pcmodel = PCMLP(0.33,alpha,betaFB=beta,gammaFw=gamma,complex_valued=True).to(device)
    checkpointPhase = torch.load(os.path.join('models',f"PCC_E19_I0_G{0.6}_B{0.2}_A{0.01}.pth"))
    pcmodel.load_state_dict(checkpointPhase["module"])
    img = torch.from_numpy(img).to(dtype=torch.float32).to(device)
    img = torch.view_as_complex(torch.stack((img[:196],img[196:]),dim=1))
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
        print('hi')
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.plot(nA[i])
        plt.subplot(1,3,2)
        plt.plot(nB[i])
        plt.subplot(1,3,3)
        plt.plot(nO[i])
        plt.savefig(os.path.join('explode_attempt_plot_norm',f'Cexpactplot_{alpha}_{i}.png'))
        plt.close()



