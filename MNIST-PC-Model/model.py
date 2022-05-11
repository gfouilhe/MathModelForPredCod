import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCMLP(nn.Module):
    '''
    Architechture :
    all FC

    784--> 120 --> 120 --> 10 
    I      A        B       O
           |---R <--|

    
    '''
    def __init__(self, memory, alphaRec, betaFB, gammaFw, num_hidden = 120):
        
        super(PCMLP,self).__init__()

        self.gammaFw = gammaFw * torch.ones(1)
        self.alphaRec = alphaRec * torch.ones(1)
        self.betaFB = betaFB * torch.ones(1)
        self.memory = memory * torch.ones(1)
        self.num_hidden = num_hidden
        self.fcin = nn.Linear(784,self.num_hidden)
        self.fcAB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBA = nn.Linear(self.num_hidden,self.num_hidden)
        #self.fcBC = nn.Linear(self.num_hidden,self.num_hidden)
        #self.fcCB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcout = nn.Linear(self.num_hidden,10)
        self.activation = F.relu

    def forward(self, i, a, b, o, networkMode):

        errorA = []
        reconstructionA = []

        batchSize = a.shape[0]

        assert networkMode in ['forward','full']

        if networkMode == 'forward':

            aNew = self.activation(self.fcin(i))
            bNew= self.activation(self.fcAB(aNew))
            #cNew = self.activation(self.fcBC(bNew))
            oNew = self.fcout(bNew)


        elif networkMode == 'full':

            if self.betaFB ==0:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = 0
            else:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.betaFB) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = torch.sigmoid(self.betaFB) / den

            errorA = nn.functional.mse_loss(self.fcBA(b), a)
            reconstructionA = torch.autograd.grad(errorA, b, retain_graph=True)[0]


            aNew = gammaFw * self.activation(self.fcin(i)) + (1 - gammaFw - betaBw) * a + betaBw * self.fcBA(b)
            bNew = gammaFw * self.activation(self.fcAB(aNew)) + + (1 - gammaFw) * b - self.alphaRec * batchSize * reconstructionA
            oNew = self.fcout(bNew)

        out =  torch.log_softmax(oNew,dim=1)


        return out, i, aNew, bNew,  oNew, reconstructionA

    