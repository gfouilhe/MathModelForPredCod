import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy
from get_data import Dataset, get_data

from network import UniDimModel

import os



betaFw = 0.6
alphaRec = 0.01
lambdaBw = 0.2


iterationNumber = 1
numberEpochs = 20
timeSteps = 30
checkpoint = [f"FF_E{numberEpochs-1}_I{it}_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth" for it in range(iterationNumber)]


def main(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps, checkpoint):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Parameters :',betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps)
        
    batchSize = 100

    dataset_train,dataset_test = get_data(10000)


    print('Train dataset size :', len(dataset_train))
    print('Test dataset size :', len(dataset_test))


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=False, num_workers=1)

    memory = 0.33

    resRecLoss = np.empty((3, numberEpochs, iterationNumber))
    resAll = np.empty((numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = UniDimModel(memory,alphaRec,betaFw,lambdaBw).to(device)
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

                a1Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                a2Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                a3Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                inputs, labels = data
               
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(dtype=torch.float32).to(device)
                optimizerPCnet.zero_grad()
                finalLoss = 0

                a1Temp.requires_grad = True
                a2Temp.requires_grad = True
                a3Temp.requires_grad = True

                outputs, a0Temp, a1Temp, a2Temp, a3Temp, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'forward')
                
                outputs, a0R, a1R, a2R, a3R, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'reconstruction')

                lossA = criterionMSE(a0Temp.view(batchSize,-1), a0R.view(batchSize,-1))
                lossB = criterionMSE(a1Temp.view(batchSize,-1), a1R.view(batchSize,-1))
                lossC = criterionMSE(a2Temp.view(batchSize,-1), a2R.view(batchSize,-1))

                finalLoss = lossA + lossB + lossC

                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            if epoch == numberEpochs - 1 :

                path = os.path.join('models',f"FFREC_E{epoch}_I{iterationIndex}_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth")
                torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test Loss/Acc
            correct = 0
            total = 0
            finalLossA = 0
            finalLossB = 0
            finalLossC = 0
            for _, data in enumerate(test_loader, 0):

                a1Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                a2Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                a3Temp = torch.randn(batchSize, 1).to(dtype=torch.float32).cuda()
                inputs, labels = data
               
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(dtype=torch.float32).to(device)

                outputs, a0Temp, a1Temp, a2Temp, a3Temp, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'forward')
             
                outputs, a0R, a1R, a2R, a3R, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'reconstruction')

                finalLossA += criterionMSE(a0Temp.view(batchSize,-1), a0R.view(batchSize,-1))
                finalLossB += criterionMSE(a1Temp.view(batchSize,-1), a1R.view(batchSize,-1))
                finalLossC += criterionMSE(a2Temp.view(batchSize,-1), a2R.view(batchSize,-1))


                predicted = torch.round(outputs)
                correct += (predicted == labels.view(batchSize,-1)).sum().item()
                total += batchSize
            acc = (100 * correct / total)
            print(acc)
            resAll[epoch, iterationIndex] = acc
            resRecLoss[0, epoch, iterationIndex] = finalLossA / total
            resRecLoss[1, epoch, iterationIndex] = finalLossB / total
            resRecLoss[2, epoch, iterationIndex] = finalLossC / total


    np.save(os.path.join('accuracies',f'REC__B{betaFw}_L{lambdaBw}_A{alphaRec}.npy'),resRecLoss)
    np.save(os.path.join('accuracies',f"ALL__B{betaFw}_L{lambdaBw}_A{alphaRec}.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs, timeSteps, checkpoint)
