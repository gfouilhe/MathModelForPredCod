from TrainAll import main as train
from TrainFFOnly import main as trainFF
from TrainRec import main as trainRec
import torch.nn.functional as F
import torch


def main():
    
    #-----------PARAMETERS--------------------
    iterationNumber = 1
    numberEpochs = 20
    timeSteps = 30
    #
    gamma_beta_couples = [(0.1,0.5),(0.05,0.5),(0.0,0.5)]#(0.7,0.1),(0.85,0.05),(0.95,0.01)] #[(gamma,beta) for gamma in gammaR for beta in betaR if beta+gamma <= 1]
    alpha_range = [0] #,0.05,0.1,0.5]
    mode = 'FFRec' #Default (All) : train everything supervised; (Rec) means unsupervised learning for reconstruction

    transpose = False
    complex_valued = False
    checkpoint = 0 #replace by list of "lalala.pth"
    activation_function = torch.tanh # or lambda x: x or F.relu

    print(f'Started train with mode {mode} with {iterationNumber} iterations, each of {numberEpochs} epochs and {timeSteps} time iterations.')
    
    if mode=='FFRec':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                trainFF(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint)
                checkpoint = [f"FF_E{numberEpochs-1}_I{it}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth" for it in range(iterationNumber)]
                trainRec(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,activation_function, transpose, complex_valued, checkpoint)

    elif mode=='FF':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                trainFF(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint)

    elif mode=='Rec':
        
        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                trainRec(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint)

        
    #############################################################################
    elif mode=='All':

        for gammaFw,betaFB in gamma_beta_couples:
            for alphaRec in alpha_range:

                train(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint)
    
    else:
        raise 'Unexpected mode'

if __name__ == "__main__":
    main()