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
numberEpochs = 15
timeSteps = 10
activation_function = F.tanh
checkpoint = 0

def main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function, checkpoint):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print('Parameters :',gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps, activation_function)
        
    batchSize = 100

    dataset_train,dataset_test = get_data()


    # print('Train dataset size :', len(dataset_train))
    # print('Test dataset size :', len(dataset_test))


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batchSize, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=False, num_workers=1)


    memory = 0.33

    resAll = np.empty((timeSteps,numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = PCMLP(memory,alphaRec,betaFB,gammaFw, activation_function).to(device)
        if checkpoint != 0:
            checkpointPhase = torch.load(os.path.join('models',checkpoint[iterationIndex]))
            pcNet.load_state_dict(checkpointPhase["module"])

        criterion = nn.CrossEntropyLoss()
        optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(0, numberEpochs):  

            print('it : ',iterationIndex)
            print('epoch : ',epoch)

            #train
            for i, data in enumerate(train_loader, 0):

                
                
                aTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                bTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                cTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                oTemp = torch.zeros(batchSize, 10).to(dtype=torch.float32).cuda()
                inputs, labels = data
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)
                optimizerPCnet.zero_grad()
                finalLoss = 0

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                cTemp.requires_grad = True
                oTemp.requires_grad = True

                outputs, iTemp, aTemp, bTemp, cTemp, oTemp, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'forward')

                loss = criterion(outputs, labels)
                finalLoss += loss
                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            
            if epoch == numberEpochs - 1 :
                path = os.path.join('models',f"FF_E{epoch}_I{iterationIndex}_G{gammaFw}_B{betaFB}_A{alphaRec}.pth")
                torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test accuracy
            correct = 0
            total = 0
            for _, data in enumerate(test_loader, 0):
                
                
                aTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                bTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                cTemp = torch.zeros(batchSize, 196).to(dtype=torch.float32).cuda()
                oTemp = torch.zeros(batchSize, 10).to(dtype=torch.float32).cuda()
                aTemp.requires_grad = True
                bTemp.requires_grad = True
                cTemp.requires_grad = True
                oTemp.requires_grad = True

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)
                outputs, iTemp, aTemp, bTemp, cTemp, oTemp, recI, recA, recB = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, cTemp, oTemp, 'forward')
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            acc = (100 * correct / total)
            print("Accuracy (ff) : ",acc)
            resAll[:, epoch, iterationIndex] = acc

    np.save(os.path.join('accuracies',f"FF__G{gammaFw}_B{betaFB}_A{alphaRec}.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main(gammaFw,betaFB,alphaRec,iterationNumber,numberEpochs,timeSteps,activation_function,checkpoint)


