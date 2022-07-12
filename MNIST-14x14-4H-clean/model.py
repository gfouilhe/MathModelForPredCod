import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexPyTorch.complexLayers import  ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

class PCMLP(nn.Module):
    '''
    Architechture :
    all FC

    196 --> 196 --> 196 --> 196 --> 10 
    I        A       B       C       D
    |---R <--|--R <--|--R <--|

    
    '''
    def __init__(self, memory, alphaRec, betaFB, gammaFw, activation_function=F.relu):
        
        super(PCMLP,self).__init__()

        self.gammaFw = gammaFw * torch.ones(1).cuda()
        self.alphaRec = alphaRec * torch.ones(1).cuda()
        self.betaFB = betaFB * torch.ones(1).cuda()
        self.memory = memory * torch.ones(1).cuda()
        self.num_hidden = 196
        self.fciA = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcAi = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcAB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBA = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBC = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcCB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcout = nn.Linear(self.num_hidden,10)
        self.activation = activation_function
        self.MSE = nn.functional.mse_loss




    def forward(self, i, a, b, c, o, networkMode):

        errorI=[]
        reconstructionI = []

        errorA = []
        reconstructionA = []

        errorB = []
        reconstructionB = []

        batchSize = a.shape[0]


        assert networkMode in ['forward','full','reconstruction']

        if networkMode == 'forward':
            aNew = self.activation(self.fciA(i))
            bNew = self.activation(self.fcAB(aNew))
            cNew = self.activation(self.fcBC(bNew))
            oNew = self.fcout(cNew)

        elif networkMode == 'reconstruction':

                i = self.fcAi(a)
                aNew = self.fcBA(b)
                bNew = self.fcCB(c)
                cNew = c
                oNew = o

        elif networkMode == 'full':

            gammaFw = self.gammaFw
            betaBw = self.betaFB
            
            errorI = self.MSE(self.fcAi(a), i)
            reconstructionI = torch.autograd.grad(errorI, a, retain_graph=True)[0]

            errorA = self.MSE(self.fcBA(b), a)
            reconstructionA = torch.autograd.grad(errorA, b, retain_graph=True)[0]

            errorB = self.MSE(self.fcCB(c), b)
            reconstructionB = torch.autograd.grad(errorB, c, retain_graph=True)[0]

            aNew = gammaFw * self.activation(self.fciA(i)) + (1 - gammaFw - betaBw) * a + betaBw * self.activation(self.fcBA(b)) - self.alphaRec * batchSize * reconstructionI
            bNew = gammaFw * self.activation(self.fcAB(aNew)) + (1 - gammaFw - betaBw) * b + betaBw * self.activation(self.fcCB(c)) - self.alphaRec * batchSize * reconstructionA
            cNew = gammaFw * self.activation(self.fcBC(bNew)) + (1 - gammaFw) * c - self.alphaRec * batchSize * reconstructionB
            oNew = self.fcout(cNew)

        out =  torch.log_softmax(oNew,dim=1)
        return out, i, aNew, bNew, cNew,  oNew, reconstructionI, reconstructionA, reconstructionB

    