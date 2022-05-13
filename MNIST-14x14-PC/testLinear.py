import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import numpy

from model import PCMLP
from getdata import get_data
import os


def main():
    
    batchSize = 100

    _,dataset_test= get_data()
    
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batchSize, shuffle=False, num_workers=1)

    gammaFw = 0.6
    alphaRec = 0.01
    betaFB = 0.2

    memory = 0.33



    iterationNumber = 5
    numberEpochs = 20
    timeSteps = 30

    resAll = np.empty((timeSteps,numberEpochs, iterationNumber))

    for iterationIndex in range(0, iterationNumber):

        for epoch in range(0, numberEpochs):

            #compute test accuracy
            pcNet = PCMLP(memory,alphaRec,betaFB,gammaFw,linear=True)
            path = os.path.join('models',f"PC_E{epoch}_I{iterationIndex}.pth")
            checkpointPhase = torch.load(path)
            pcNet.load_state_dict(checkpointPhase["module"])
            correct = np.zeros(timeSteps)
            total = 0

            for _, data in enumerate(test_loader, 0):

                aTemp = torch.zeros(batchSize, 120)
                bTemp = torch.zeros(batchSize, 120)
                oTemp = torch.zeros(batchSize, 10)

                aTemp.requires_grad = True
                bTemp.requires_grad = True
                oTemp.requires_grad = True

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'forward')
                for tt in range(timeSteps):
                    outputs, iTemp, aTemp, bTemp, oTemp, reconstruction = pcNet(inputs.view(batchSize,-1), aTemp, bTemp, oTemp, 'full')

                    _, predicted = torch.max(outputs.data, 1)
                    correct[tt] = correct[tt] + (predicted == labels).sum().item()

                total += labels.size(0)

            resAll[:, epoch, iterationIndex] = (100 * correct / total)

    np.save(os.path.join('accuracies',f"accLinear.npy"), resAll)
    print('Finished Training')

if __name__ == "__main__":
    main()

