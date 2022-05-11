import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy

from model import PCMLP
from data import Dataset, get_data

import os


def main():
        
    batchSize = 32


    X_train, X_test, y_train, y_test = get_data(n=2048,std=0.1,scale_factor=0.85)
    X_train, X_test, y_train, y_test = torch.from_numpy(X_train.astype(numpy.float32)), torch.from_numpy(X_test.astype(numpy.float32)), \
                                        torch.from_numpy(y_train.astype(numpy.float32)), torch.from_numpy(y_test.astype(numpy.float32))

    dataset_train = Dataset(X_train,y_train)
    dataset_test = Dataset(X_test,y_test)

    batch_size=32

    print('Train dataset size :', X_train.shape[0])
    print('Test dataset size :', X_test.shape[0])


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
    classes = ('0', '1') 
    featuresA = 5
    featuresB = 5
    featureso = 2

    gammaFw = 0.5
    alphaRec = 0.01
    betaFB = 0.2

    memory = 0.33



    iterationNumber = 10
    numberEpochs = 30
    timeSteps = 10

    resAll = np.empty((timeSteps,numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = PCMLP(featuresA,featuresB,memory,alphaRec,betaFB,gammaFw)


        criterion = nn.BCELoss()
        optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.005, momentum=0.9)

        for epoch in range(0, numberEpochs):  

            print('it : ',iterationIndex)
            print('epoch : ',epoch)

            #train
            for i, data in enumerate(train_loader, 0):

                aTemp = torch.zeros(batchSize, featuresA)
                bTemp = torch.zeros(batchSize, featuresB)
                oTemp = torch.zeros(batchSize, 1)


                inputs, labels = data

                optimizerPCnet.zero_grad()
                finalLoss = 0

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs, aTemp, bTemp, oTemp, 'forward')

                for tt in range(timeSteps):
                    outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs, aTemp, bTemp, oTemp, 'full')
                    loss = criterion(outputs, labels)
                    finalLoss += loss

                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            path = os.path.join('models',f"PC_E{epoch}_I{iterationIndex}.pth")
            torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            #compute test accuracy
            correct = np.zeros(timeSteps)
            total = 0
            for _, data in enumerate(test_loader, 0):

                aTemp = torch.zeros(batchSize, featuresA)
                bTemp = torch.zeros(batchSize, featuresB)
                oTemp = torch.zeros(batchSize, 1)

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs, aTemp, bTemp, oTemp, 'forward')
                for tt in range(timeSteps):
                    outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs, aTemp, bTemp, oTemp, 'full')

                    predicted = outputs.reshape(-1).detach().round()
                    correct[tt] = correct[tt] + (predicted == labels).sum().item()

                total += labels.size(0)

            resAll[:, epoch, iterationIndex] = (100 * correct / total)

    np.save(os.path.join('accuracies',f"accTrainingCE.npy"), resAll)
    print('Finished Training')


if __name__ == "__main__":
    main()


