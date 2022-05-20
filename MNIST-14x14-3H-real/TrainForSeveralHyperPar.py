from TrainAll import main as train

def main():
    
    iterationNumber = 1
    numberEpochs = 20
    timeSteps = 30

    gamma_beta_couples = [(0.1,0.7),(0.25,0.5),(0.33,0.33),(0.5,0.25),(0.7,0.1)]
    alpha_range = [0.01,0.05,0.1,0.5]

    for gammaFw,betaFB in gamma_beta_couples:
        for alphaRec in alpha_range:
            train(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps)

if __name__ == "__main__":
    main()