from math import gamma
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy

from model import PCMLP
from utils.getdata import get_data

import os


gammaFw = 0.6
alphaRec = 0.01
betaFB = 0.2


iterationNumber = 1
numberEpochs = 20
timeSteps = 30
activation_function = F.tanh
transpose = False
complex_valued = False
checkpoint = [f"FF_E{numberEpochs-1}_I{it}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth" for it in range(iterationNumber)]


def main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, transpose, complex_valued, checkpoint):

    assert checkpoint != 0, 'Reconstuction should be used with a checkpoint from FF learned weights'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Parameters :',gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps,activation_function, transpose, complex_valued)
        
    batchSize = 100

    dataset_train,dataset_test = get_data()


    # print('Train dataset size :', len(dataset_train))
    # print('Test dataset size :', len(dataset_test))


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=False, num_workers=1)


    memory = 0.33

    resAll = np.empty((timeSteps,numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = PCMLP(memory,alphaRec,betaFB,gammaFw,transpose,complex_valued,activation_function).to(device)
        if checkpoint != 0:
            checkpointPhase = torch.load(os.path.join('models',checkpoint[iterationIndex]))
            pcNet.load_state_dict(checkpointPhase["module"])

        criterionMSE = nn.functional.mse_loss
        optimizerPCnet = optim.Adam(pcNet.parameters(), lr=0.001)

        for epoch in range(0, numberEpochs):  

            print('it : ',iterationIndex)
            print('epoch : ',epoch)

            #train
            for i, data in enumerate(train_loader, 0):

                if complex_valued:
                    aTemp = torch.zeros(batchSize, 196).to(dtype=torch.complex64).cuda()
                    bTemp = torch.zeros(batchSize, 196).to(dtype=torch.complex64).cuda()
                    oTemp = torch.zeros(batchSize, 10).to(dtype=torch.complex64).cuda()
                else:
                    aTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    bTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    oTemp = torch.zeros(batchSize, 10).to(dtype=torch.float32).cuda()

                inputs, labels = data
                if complex_valued:
                    inputs, labels = inputs.to(dtype=torch.complex64).to(device),labels.to(device)
                else:
                    inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)

                optimizerPCnet.zero_grad()
                finalLoss = 0

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
                outputs, iR, aR, bR, oR, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')

                lossA = criterionMSE(inputs.view(batchSize,-1), iR)
                lossB = criterionMSE(aTemp.view(batchSize,-1), aR)

                finalLoss = lossA + lossB

                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            if epoch == numberEpochs - 1 :

                path = os.path.join('models',f"FFREC_E{epoch}_I{iterationIndex}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth")
                torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test accuracy
            correct = np.zeros(timeSteps)
            total = 0
            for _, data in enumerate(test_loader, 0):

                if complex_valued:
                    aTemp = torch.zeros(batchSize, 196).to(dtype=torch.complex64).cuda()
                    bTemp = torch.zeros(batchSize, 196).to(dtype=torch.complex64).cuda()
                    oTemp = torch.zeros(batchSize, 10).to(dtype=torch.complex64).cuda()
                else:
                    aTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    bTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                    oTemp = torch.zeros(batchSize, 10).to(dtype=torch.float32).cuda()

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                if complex_valued:
                    inputs, labels = inputs.to(dtype=torch.complex64).to(device),labels.to(device)
                else:
                    inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)

                outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
                for tt in range(timeSteps):
                    outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')

                    _, predicted = torch.max(outputs.data, 1)
                    correct[tt] = correct[tt] + (predicted == labels).sum().item()

                total += labels.size(0)

            resAll[:, epoch, iterationIndex] = (100 * correct / total)

    np.save(os.path.join('accuracies',f"ALL__G{gammaFw}_B{betaFB}_A{alphaRec}.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps,activation_function, transpose, complex_valued, checkpoint)


