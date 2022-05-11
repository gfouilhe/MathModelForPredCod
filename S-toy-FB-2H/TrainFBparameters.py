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



    iterationNumber = 10
    numberEpochs = 30

    resRecLossAll = np.empty((numberEpochs, iterationNumber))
    resRecAll = np.empty((numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        print(iterationIndex)

        pcNet = PCMLP(featuresA,featuresB,memory,alphaRec,betaFB,gammaFw)

        checkpointPhase = torch.load(os.path.join('models',f"PC_FF_E29_I{iterationIndex}.pth"))
        pcNet.load_state_dict(checkpointPhase["module"])

        for name, p in pcNet.named_parameters():
            if name.split('.')[0] in ['fcin', 'fcAB', 'fcBA', 'fcout']:
                p.requires_grad_(False)

        criterionMSE = nn.functional.mse_loss
        optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

        for epoch in range(0, numberEpochs):  

            for _, data in enumerate(train_loader, 0):

                a = torch.randn(batchSize, featuresA)
                b = torch.randn(batchSize, featuresB)
                o = torch.randn(batchSize, 1)

                inputs, labels = data

                optimizerPCnet.zero_grad()
                lossRec = 0

                outputs, i, a, b, o, reconstruction = pcNet(inputs, a, b, o, 'forward')
                outputs, iR, aR, bR, oR, reconstruction = pcNet(inputs, a, b, o, 'reconstruction')

                lossRec = criterionMSE(a, aR)
                lossRec.requires_grad = True

                lossRec.backward(retain_graph=True) 
                optimizerPCnet.step()

            path = os.path.join('models',f"PC_FF_FB_E{epoch}_I{iterationIndex}.pth")
            torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

            finalLossRec = 0

            correct = 0
            total = 0

            for _, data in enumerate(test_loader, 0):

                a = torch.randn(batchSize, featuresA)
                b = torch.randn(batchSize, featuresB)
                o = torch.randn(batchSize, 1)

                inputs, labels = data
                inputs = inputs
                labels = labels

                outputs, i, a, b, o, reconstruction = pcNet(inputs, a, b, o, 'forward')
                outputs, iR, aR, bR, oR, reconstruction = pcNet(inputs, a, b, o, 'reconstruction')

                finalLossRec += criterionMSE(a, aR)

                total += labels.size(0)

                predicted = outputs.reshape(-1).detach().round()
                correct += (predicted == labels).sum().item()

            resRecAll[epoch, iterationIndex] = (100 * correct / total)

            resRecLossAll[epoch, iterationIndex] = finalLossRec / total

            print(100 * correct / total)
            print(finalLossRec / total)

    print('Finished Training')
    np.save(os.path.join('accuracies',f"recLossTrainingRECff_rec.npy"), resRecLossAll)
    np.save(os.path.join('accuracies',f"accTrainingRECff_rec.npy"), resRecAll)


if __name__ == "__main__":
    main()

