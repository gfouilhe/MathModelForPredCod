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


gammaFw = 0.1
alphaRec = 0.01
betaFB = 0.8


iterationNumber = 1
numberEpochs = 20
activation_function = torch.tanh
checkpoint = [f"FF_E{numberEpochs-1}_I{it}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth" for it in range(iterationNumber)]


def main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs, activation_function, checkpoint):

    assert checkpoint != 0, 'Reconstuction should be used with a checkpoint from FF learned weights'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Parameters :',gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,activation_function)
        
    batchSize = 100
    dataset_train,dataset_test = get_data()
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=False, num_workers=1)

    memory = 0.33

    resRecLoss = np.empty((3, numberEpochs, iterationNumber))
    resAll = np.empty((numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = PCMLP(memory,alphaRec,betaFB,gammaFw,activation_function).to(device)
        if checkpoint != 0:
            checkpointPhase = torch.load(os.path.join('models',checkpoint[iterationIndex]))
            pcNet.load_state_dict(checkpointPhase["module"])

        criterionMSE = nn.functional.mse_loss
        criterionCE = nn.CrossEntropyLoss()
        optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(0, numberEpochs):  

            print('it : ',iterationIndex)
            print('epoch : ',epoch)

            #train
            for i, data in enumerate(train_loader, 0):
                
                aTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                bTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                cTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                oTemp = torch.randn(batchSize, 10).to(dtype=torch.float32).cuda()

                inputs, labels = data
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)

                optimizerPCnet.zero_grad()
                finalLoss = 0

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                cTemp.requires_grad = True
                oTemp.requires_grad = True

                outputs, iTemp, aTemp, bTemp, cTemp, oTemp, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'forward')
                outputs, iR, aR, bR, cR, oR, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'reconstruction')

                lossA = criterionMSE(inputs.view(batchSize,-1), iR)
                lossB = criterionMSE(aTemp.view(batchSize,-1), aR)
                lossC = criterionMSE(bTemp.view(batchSize,-1), bR)
                lossAcc = criterionCE(outputs,labels)

                finalLoss = lossA + lossB + lossC + 0.5*lossAcc
                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            if epoch == numberEpochs - 1 :

                path = os.path.join('models',f"FFREC_E{epoch}_I{iterationIndex}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth")
                torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test Loss/Acc
            correct = 0
            total = 0
            finalLossA = 0
            finalLossB = 0
            finalLossC = 0
            finalLossAcc = 0
            for _, data in enumerate(test_loader, 0):

                aTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                bTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                cTemp = torch.randn(batchSize, 196).to(dtype=torch.float32).cuda()
                oTemp = torch.randn(batchSize, 10).to(dtype=torch.float32).cuda()

                inputs, labels = data
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                cTemp.requires_grad = True
                oTemp.requires_grad = True

                outputs, iTemp, aTemp, bTemp, cTemp, oTemp, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'forward')
                outputs, iR, aR, bR, cR, oR, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'reconstruction')


                finalLossA += criterionMSE(inputs.view(batchSize,-1), iR)
                finalLossB += criterionMSE(aTemp.view(batchSize,-1), aR)
                finalLossC += criterionMSE(bTemp.view(batchSize,-1), bR)
                finalLossAcc += criterionCE(outputs,labels)

                _, predicted = torch.max(outputs.data, 1)
                correct +=  (predicted == labels).sum().item()
                
                total += labels.size(0)
            acc = (100 * correct / total)
            print('Accuracy (fb) : ', acc)
            resAll[epoch, iterationIndex] = acc
            resRecLoss[0, epoch, iterationIndex] = finalLossA / total
            resRecLoss[1, epoch, iterationIndex] = finalLossB / total
            resRecLoss[2, epoch, iterationIndex] = finalLossC / total


    np.save(os.path.join('accuracies',f'REC__G{gammaFw}_B{betaFB}_A{alphaRec}.npy'),resRecLoss)
    np.save(os.path.join('accuracies',f"ALL__G{gammaFw}_B{betaFB}_A{alphaRec}.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,activation_function, checkpoint)
