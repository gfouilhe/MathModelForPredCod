# from TrainAll import main as train
from TrainFFOnly import main as trainFF
from TrainRec import main as trainRec
import torch.nn.functional as F
import torch


def main():
    
    #-----------PARAMETERS--------------------
    iterationNumber = 1
    numberEpochs = 20
    timeSteps = 30

    parameters = [(0.6,0.2,0.01),(0.33,0.33,0.01),(0.5,0.25,0.01),(0.7,0.1,0.01),(0.85,0.05,0.01),(0.95,0.01,0.01)]
    mode = 'FFRec' #Default (All) : train everything supervised; (Rec) means unsupervised learning for reconstruction


    checkpoint = 0 #replace by list of "lalala.pth"

    print(f'Started train with mode {mode} with {iterationNumber} iterations, each of {numberEpochs} epochs and {timeSteps} time iterations.')
    
    if mode=='FFRec':

        for betaFw,lambdaBw,alphaRec in parameters:
            
            trainFF(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps, checkpoint)
            checkpoint = [f"FF_E{numberEpochs-1}_I{it}_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth" for it in range(iterationNumber)]
            trainRec(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps, checkpoint)

    elif mode=='FF':

        for betaFw,lambdaBw,alphaRec in parameters:
            
            trainFF(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps, checkpoint)

    elif mode=='Rec':
        
        for betaFw,lambdaBw,alphaRec in parameters:
            
            trainRec(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps, checkpoint)

        
    #############################################################################
    # elif mode=='All':

    #     for gammaFw,betaFB in gamma_beta_couples:
    #         for alphaRec in alpha_range:

    #             train(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint)
    
    else:
        raise 'Unexpected mode'

if __name__ == "__main__":
    main()