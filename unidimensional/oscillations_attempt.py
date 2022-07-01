import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from network import UniDimModel
from PIL import Image
import pickle

batchSize = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def main():

    #------ Parameteters -------

    mode = 'oscillations' # 'explode' , 'dump', 'oscillations'

    displaymode ='neurons' # 'neurons', 'layernorm'

     
    long='' # 'long' = 200 iterations instead of 50 ('')

    UsedForLearningHyper = [(0.6,0.2,0.01),(0.33,0.33,0.01),(0.5,0.25,0.01),(0.7,0.1,0.01),(0.85,0.05,0.01),(0.95,0.01,0.01)]
    comment = ''
    alphaR = [0.01]#,0.05,0.1,0.25]
    numberEpochs = 20
    timeSteps = 50
    commentact = 'linear' #'tanh' ; 'relu' 'linear'
    
    if commentact == 'linear' :
        activation = lambda x: x 
    elif commentact == 'tanh':
        activation = torch.tanh
    elif commentact == 'relu':
        activation = torch.relu

    for betaFw, lambdaBw, alphaRec in UsedForLearningHyper:

        for alpha in alphaR:


            if mode=='oscillations':

                path = os.path.join('parameters_setup',f'{comment}_good_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl')
                

            if mode=='explode':

                path = os.path.join('parameters_setup',f'{comment}_over_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl')
                
            if mode=='dump':
                
                path = os.path.join('parameters_setup',f'{comment}_under_params_dictionary_B{betaFw}_L{lambdaBw}_A{alphaRec}_{alpha}.pkl')

            with open(path,'rb') as f:
                params_and_imgs = pickle.load(f)

            params_list = [param for _,param in params_and_imgs.items()]
            params_list = params_list[:20]

            act0 = np.zeros((len(params_list),timeSteps))
            act1 = np.zeros((len(params_list),timeSteps))
            act2 = np.zeros((len(params_list),timeSteps))
            act3 = np.zeros((len(params_list),timeSteps))


            for i,im in enumerate(params_list):
                lamb,beta,img = im
                a1Temp = img[0]
                a1Temp = torch.tensor(a1Temp.astype('float32')).to(device).view(batchSize,-1)
                a0Temp = torch.clone(a1Temp)
                a2Temp = img[1]
                a2Temp = torch.tensor(a2Temp.astype('float32')).to(device).view(batchSize,-1)
                a3Temp = img[2]
                a3Temp = torch.tensor(a3Temp.astype('float32')).to(device).view(batchSize,-1)
                pcmodel = UniDimModel(0.33,alphaRec,betaFw,lambdaBw,activation_function=activation).to(device)
                checkpointPhase = torch.load(os.path.join('models',f"FFREC_E{numberEpochs-1}_I0_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth"))
                pcmodel.load_state_dict(checkpointPhase["module"])

                a0Temp.requires_grad = True
                a1Temp.requires_grad = True
                a2Temp.requires_grad = True
                a3Temp.requires_grad = True

                _, a0Temp, _, _, _, _,_ ,_ = pcmodel(a0Temp, a1Temp, a2Temp, a3Temp, 'reconstruction')

                # actA[i,0,:] = aTemp.detach().cpu().numpy()
                # actB[i,0,:] = bTemp.detach().cpu().numpy()
                # actO[i,0,:] = oTemp.detach().cpu().numpy()

                for t in range(timeSteps):
                    
                    _, a0Temp, a1Temp, a2Temp, a3Temp, _, _, _ = pcmodel(a0Temp, a1Temp, a2Temp, a3Temp, 'full')
                    act0[i,t] = a0Temp.detach().cpu().numpy()
                    act1[i,t] = a1Temp.detach().cpu().numpy()
                    act2[i,t] = a2Temp.detach().cpu().numpy()
                    act3[i,t] = a3Temp.detach().cpu().numpy()
            
            

            if displaymode=='neurons':


                for i, (beta,lamb,_) in enumerate(params_list):

                    if np.size(act0[i,:])==0:
                        pass
                    else:
                                
                        plt.figure(figsize=(15,5))
                        plt.subplot(1,4,1)
                        plt.plot(act0[i,:])
                        plt.title(f'Layer 0')
                        plt.subplot(1,4,2)
                        plt.plot(act1[i,:])
                        plt.title(f'Layer 1')
                        plt.subplot(1,4,3)
                        plt.plot(act2[i,:])
                        plt.title(f'Layer 2')
                        plt.subplot(1,4,4)
                        plt.plot(act3[i,:])
                        plt.title(f'Layer 3')
                        plt.savefig(os.path.join(f'{mode}_attempt_plot',f'{displaymode}{long}{commentact}actplot_B{betaFw}_L{lambdaBw}_A{alphaRec}_B{beta}_L{lamb}_A{alpha}_{i}.png'))
                        plt.close()
if __name__ == "__main__":
    main()