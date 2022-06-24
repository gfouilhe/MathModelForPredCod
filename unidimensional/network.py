import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class UniDimModel(nn.Module):

    def __init__(self, memory, alphaRec, betaFw, lambdaBw, activation_function=torch.tanh):

        super(UniDimModel,self).__init__()

        self.lambdaBw = lambdaBw * torch.ones(1).cuda()
        self.alphaRec = alphaRec * torch.ones(1).cuda()
        self.betaFw = betaFw * torch.ones(1).cuda()
        self.memory = memory * torch.ones(1).cuda()
        self.fc01 = nn.Linear(1,1)
        self.fc10 = nn.Linear(1,1)
        self.fc12 = nn.Linear(1,1)
        self.fc21 = nn.Linear(1,1)
        self.fc23 = nn.Linear(1,1)
        self.fc32 = nn.Linear(1,1)
        self.activation = activation_function
        self.MSE = nn.functional.mse_loss

    def forward(self, a0, a1, a2, a3, mode='forward'):

        batchSize = a0.shape[0]

        assert mode in ['forward','reconstruction','full']

        reconstruction0, reconstruction1, reconstruction2 = (1,2,3)

        if mode=='forward':
            
            a1 = self.activation(self.fc01(a0))
            a2 = self.activation(self.fc12(a1))
            a3 = self.activation(self.fc23(a2))

            

        elif mode=='reconstruction':
            
            a0 = self.activation(self.fc10(a1))
            a1 = self.activation(self.fc21(a2))
            a2 = self.activation(self.fc32(a3))

        elif mode=='full':

            bFw = self.betaFw
            lBw = self.lambdaBw
            aRec = self.alphaRec

            error0 = self.MSE(self.fc10(a1), a0)
            reconstruction0 = torch.autograd.grad(error0, a1, retain_graph=True)[0]

            error1 = self.MSE(self.fc21(a2), a1)
            reconstruction1 = torch.autograd.grad(error1, a2, retain_graph=True)[0]

            error2 = self.MSE(self.fc32(a3), a2)
            reconstruction2 = torch.autograd.grad(error2, a3, retain_graph=True)[0]

            a1 = bFw * self.activation(self.fc01(a0)) + (1 - bFw - lBw) * a1 + lBw * self.activation(self.fc21(a2)) - aRec * batchSize * reconstruction0
            a2 = bFw * self.activation(self.fc12(a1)) + (1 - bFw - lBw) * a2 + lBw * self.activation(self.fc32(a3)) - aRec * batchSize * reconstruction1
            a3 = bFw * self.activation(self.fc23(a2)) + (1 - bFw) * a3 - aRec * batchSize * reconstruction2
        
        output = torch.sigmoid(a3)

        return output, a0, a1, a2, a3, reconstruction0, reconstruction1, reconstruction2