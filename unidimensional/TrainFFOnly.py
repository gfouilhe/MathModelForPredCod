from math import gamma
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
checkpoint = 0


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

    resAll = np.empty((timeSteps,numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = UniDimModel(memory,alphaRec,betaFw,lambdaBw).to(device)
        if checkpoint != 0:
            checkpointPhase = torch.load(os.path.join('models',checkpoint[iterationIndex]))
            pcNet.load_state_dict(checkpointPhase["module"])

        criterion = nn.BCELoss() # nn.CrossEntropyLoss() 
        optimizerPCnet = optim.Adam(pcNet.parameters(), lr=0.001)

        for epoch in range(0, numberEpochs):  

            print('it : ',iterationIndex)
            print('epoch : ',epoch)

            #train
            for i, data in enumerate(train_loader, 0):

                
                a1Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                a2Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                a3Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                inputs, labels = data
               
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)
                optimizerPCnet.zero_grad()
                finalLoss = 0

                a1Temp.requires_grad = True
                a2Temp.requires_grad = True
                a3Temp.requires_grad = True

                outputs, a0Temp, a1Temp, a2Temp, a3Temp, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'forward')

                loss = criterion(outputs.view(batchSize,-1), labels.view(batchSize,-1))
                finalLoss += loss
                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            
            if epoch == numberEpochs - 1 :
                path = os.path.join('models',f"FF_E{epoch}_I{iterationIndex}_B{betaFw}_L{lambdaBw}_A{alphaRec}.pth")
                torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test accuracy
            correct = 0
            total = 0
            for _, data in enumerate(test_loader, 0):
                
                a1Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                a2Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                a3Temp = torch.zeros(batchSize, 1).to(dtype=torch.float32).cuda()
                inputs, labels = data
               
                inputs, labels = inputs.to(dtype=torch.float32).to(device),labels.to(device)

                a1Temp.requires_grad = True
                a2Temp.requires_grad = True
                a3Temp.requires_grad = True
               
                outputs, a0Temp, a1Temp, a2Temp, a3Temp, reconstruction0, reconstruction1, reconstruction2 = pcNet(inputs.view(batchSize,-1), a1Temp, a2Temp, a3Temp, 'forward')
                
                _, predicted = torch.max(outputs.data, 1)
                correct+= (predicted == labels).sum().item()
                total += labels.size(0)

            resAll[:, epoch, iterationIndex] = (100 * correct / total)

    np.save(os.path.join('accuracies',f"FF__B{betaFw}_L{lambdaBw}_A{alphaRec}.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main(betaFw,lambdaBw,alphaRec,iterationNumber,numberEpochs,timeSteps,checkpoint)


