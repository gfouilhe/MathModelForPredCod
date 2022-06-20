import torch
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

    gammaFw = 0.33
    alphaRec = 0.01
    betaFB = 0.33

    memory = 0

    #first FF pass
    iterationNumber = 10
    numberEpochs = 30

    resFFall = np.empty((numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        pcNet = PCMLP(featuresA,featuresB,memory,alphaRec,betaFB,gammaFw)

        criterion = nn.BCELoss()
        optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(0, numberEpochs):  
            print('it = ', iterationIndex)
            print('epoch = ', epoch)

            for _ , data in enumerate(train_loader, 0):

                a = torch.randn(batchSize, featuresA)
                b = torch.randn(batchSize, featuresB)
                o = torch.randn(batchSize, 1)

                inputs, labels = data
                inputs = inputs
                labels = labels

                optimizerPCnet.zero_grad()
                finalLoss = 0

                outputs, i, a, b, o, reconstruction = pcNet(inputs, a, b, o, 'forward')
                finalLoss = criterion(outputs, labels)

                finalLoss.backward(retain_graph=True)  
                optimizerPCnet.step()

            path = os.path.join('models',f"PC_FF_E{epoch}_I{iterationIndex}.pth")
            torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            correct = 0
            total = 0

            for _ , data in enumerate(test_loader, 0):

                a = torch.randn(batchSize, featuresA)
                b = torch.randn(batchSize, featuresB)
                o = torch.randn(batchSize, 1)

                inputs, labels = data
                inputs = inputs
                labels = labels

                outputs, i, a, b, o, reconstruction = pcNet(inputs, a, b, o, 'forward')

                predicted = outputs.reshape(-1).detach().round()
                correct += (predicted == labels).sum().item()

                total += labels.size(0)

            resFFall[epoch, iterationIndex] = (100 * correct / total)

    np.save(os.path.join('accuracies',f"accTrainingRECff.npy"), resFFall)

if __name__ == "__main__":
    main()

