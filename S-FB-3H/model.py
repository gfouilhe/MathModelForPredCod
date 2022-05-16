import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCMLP(nn.Module):
    '''
    Architechture :
    all FC

    2 --> 5 --> 5 --> 5 --> 1 
    I     A     B     C     O
          |-R <-||-R <-|

    
    '''
    def __init__(self, featA, featB, featC, memory, alphaRec, betaFB, gammaFw, num_hidden = 5):
        
        super(PCMLP,self).__init__()

        self.gammaFw = gammaFw * torch.ones(1)
        self.alphaRec = alphaRec * torch.ones(1)
        self.betaFB = betaFB * torch.ones(1)
        self.memory = memory * torch.ones(1)
        self.num_hidden = num_hidden
        self.fcin = nn.Linear(2,self.num_hidden)
        self.fcAB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBA = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBC = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcCB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcout = nn.Linear(self.num_hidden,1)

    def forward(self, i, a, b, c, o, networkMode):

        errorA = []
        reconstructionA = []

        batchSize = a.shape[0]

        assert networkMode in ['forward','reconstruction','full']

        if networkMode == 'forward':

            aNew = F.relu(self.fcin(i))
            bNew= F.relu(self.fcAB(aNew))
            cNew = F.relu(self.fcBC(bNew))
            oNew = self.fcout(cNew)
            reconstructionA = None
            reconstructionB = None
            

        elif networkMode == 'reconstruction':

            aNew = self.fcBA(b)
            bNew = self.fcCB(c)
            cNew = c
            oNew = o
            reconstructionA = aNew
            reconstructionB = bNew

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

            errorB = nn.functional.mse_loss(self.fcBA(c), b)
            reconstructionB = torch.autograd.grad(errorB, c, retain_graph=True)[0]

            aNew = gammaFw * F.relu(self.fcin(i)) + (1 - gammaFw - betaBw) * a + betaBw * self.fcBA(b)
            bNew = gammaFw * F.relu(self.fcAB(aNew)) + + (1 - gammaFw - betaBw) * b + betaBw * self.fcCB(c) - self.alphaRec * batchSize * reconstructionA
            cNew = gammaFw * F.relu(self.fcBC(bNew)) + (1 - gammaFw) * c - self.alphaRec * batchSize * reconstructionB
            oNew = self.fcout(cNew)

        out =  torch.sigmoid(torch.flatten(oNew))


        return out, i, aNew, bNew, cNew, oNew, reconstructionA, reconstructionB

    