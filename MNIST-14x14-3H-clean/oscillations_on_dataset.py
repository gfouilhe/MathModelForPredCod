import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle
from utils.getdata import get_data


batchSize = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():

    #------ Parameteters -------

    displaymode ='neurons' # 'neurons', 'layernorm'
     
    long='' # 'long' = 200 time iterations instead of 50 ('')

    UsedForLearningHyper =  [(0.33,0.33,100)]#,(0.85,0.05,0.01),(0.95,0.01,0.01),(0.7,0.1,0.01)]
    comment = ''
    alphaR = [60] #list(np.arange(80,120,5))
    numberEpochs = 10
    timeSteps = 200
    commentact = 'tanh' #'tanh' ; 'relu' 'linear'
    
    if commentact == 'linear' :
        activation = lambda x: x 
    elif commentact == 'tanh':
        activation = torch.tanh
    elif commentact == 'relu':
        activation = torch.relu
    
    _, dataset = get_data()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=1)
    

    for gammaFw, betaFB, alphaRec in UsedForLearningHyper:

        for alpha in alphaR:

            path = os.path.join('parameters_setup',f'{comment}_good_params_dictionary_G{gammaFw}_B{betaFB}_A{alphaRec}.pkl')

            with open(path,'rb') as f:
                params_and_imgs = pickle.load(f)

            params_list = [param for _,param in params_and_imgs.items()]
            # params_list = params_list[:20]
            params_list = [(0.33,0.33,0)]
            
            k = 0

            for i,params in enumerate(params_list):

                gamma,beta,_ = params

                for j,data in enumerate(data_loader):
                    print('new img, ', alpha)

                    if j>10:
                        break

                    k+=1

                    actA = np.zeros((timeSteps,196))
                    actB = np.zeros((timeSteps,196))
                    actO = np.zeros((timeSteps,10))

                    img, label = data
                    iTemp = img.to(dtype=torch.float32).to(device)
                    aTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    bTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    oTemp = torch.zeros(batchSize, 10).to(dtype=torch.float32).cuda()
                    pcmodel = PCMLP(0.33,alphaRec=alpha,betaFB=beta,gammaFw=gamma,activation_function=activation).to(device)
                    checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_G{gammaFw}_B{betaFB}_A{alphaRec}.pth"))
                    pcmodel.load_state_dict(checkpointPhase["module"])

                    outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcmodel(iTemp.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
                    _, predicted = torch.max(outputs.data, 1)

                    print('label : ', label)
                    print('output after first ff pass : ', predicted)
                    
                    for t in range(timeSteps):
                        
                        outputs, _, aTemp, bTemp, oTemp, _ =  pcmodel(iTemp.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')

                        if t%33==0:
                            _, predicted = torch.max(outputs.data, 1)
                            print(f'output  (t = {t}): ', predicted)

                        actA[t,:] = aTemp.detach().cpu().numpy()
                        actB[t,:] = bTemp.detach().cpu().numpy()
                        actO[t,:] = oTemp.detach().cpu().numpy()  


                    if np.size(actA[:,:])==0:
                        pass
                    else:
                        for j in range(5):
                                
                            plt.figure(figsize=(15,5))
                            plt.subplot(1,2,1)
                            plt.plot(actA[:,j])
                            # plt.yscale('log')
                            plt.title(f'Layer A, neuron {j}')
                            plt.subplot(1,2,2)
                            plt.plot(actB[:,j])
                            # plt.yscale('log')
                            plt.title(f'Layer B, neuron {j}')
                            plt.savefig(os.path.join(f'activations_on_data',f'{displaymode}{long}{commentact}actplot_G{gammaFw}_B{betaFB}_A{alphaRec}_G{gamma}_B{beta}_A{alpha}_{k}.png'))
                            plt.close()
if __name__ == "__main__":
    main()