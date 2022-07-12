from TrainAll import main as train
from TrainFFOnly import main as trainFF
from TrainRec import main as trainRec
import torch.nn.functional as F
import torch


def main():
    
    #-----------PARAMETERS--------------------
    iterationNumber = 1
    numberEpochs = 15
    timeSteps = 10
    #
    gamma_beta_couples = [(0.1,0.8), (0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]
    alpha_range = [0.01] #,0.05,0.1,0.5]
    mode = 'FFRec' #Default (All) : train everything supervised; (Rec) means unsupervised learning for reconstruction
    already_trained_ff = []# for alpha = 0.01 :(0.1,0.8), (0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]
    already_trained_fb = []# for alpha = 0.01 :(0.1,0.8), (0.1,0.5), (0.1,0.1),(0.33,0.33),(0.2,0.5),(0.5,0.2),(0.5,0.1),(0.8,0.1)]
    checkpoint = 0 #replace by list of "lalala.pth"
    activation_function = torch.tanh # or lambda x: x or F.relu

    print(f'Started train with mode {mode} with {iterationNumber} iterations, each of {numberEpochs} epochs and {timeSteps} time iterations.')
    
    if mode=='FFRec':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:
                
                if (gammaFw,betaFB) not in already_trained_ff:
                    trainFF(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, checkpoint)
                checkpoint = [f"FF_E{numberEpochs-1}_I{it}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth" for it in range(iterationNumber)]
                if (gammaFw,betaFB) not in already_trained_fb:
                    trainRec(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,activation_function, checkpoint)

    elif mode=='FF':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                if (gammaFw,betaFB) not in already_trained_ff:
                    trainFF(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, checkpoint)

    elif mode=='Rec':
        
        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                if (gammaFw,betaFB) not in already_trained_fb:
                    trainRec(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, checkpoint)

        
    #############################################################################
    elif mode=='All':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                train(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, checkpoint)
    
    else:
        raise 'Unexpected mode'

if __name__ == "__main__":
    main()