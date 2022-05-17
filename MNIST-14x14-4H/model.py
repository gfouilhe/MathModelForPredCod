import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCMLP(nn.Module):
    '''
    Architechture :
    all FC

    196--> 196 --> 196 --> 196 --> 196 --> 10 
    I      A        B       C       D       O
           |---R <--|--R <--|--R <--|

    
    '''
    def __init__(self, memory, alphaRec, betaFB, gammaFw,linear=False):
        
        super(PCMLP,self).__init__()

        self.gammaFw = gammaFw * torch.ones(1)
        self.alphaRec = alphaRec * torch.ones(1)
        self.betaFB = betaFB * torch.ones(1)
        self.memory = memory * torch.ones(1)
        self.num_hidden = 196
        self.fcin = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcAB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBA = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcBC = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcCB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcCD = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcDC = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcout = nn.Linear(self.num_hidden,10)
        self.activation = F.relu
        if linear:
            self.activation = lambda x: x

    def forward(self, i, a, b, c, d, o, networkMode):

        reconstructionA = []
        reconstructionB = []
        reconstructionC = []

        batchSize = a.shape[0]


        assert networkMode in ['forward','full']

        if networkMode == 'forward':

            aNew = self.activation(self.fcin(i))
            bNew = self.activation(self.fcAB(aNew))
            cNew = self.activation(self.fcBC(bNew))
            dNew = self.activation(self.fcCD(cNew))
            oNew = self.fcout(dNew)


        elif networkMode == 'full':

            if self.betaFB ==0:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = 0
            else:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.betaFB) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = torch.sigmoid(self.betaFB) / den



            a = self.activation(a)
            b = self.activation(b)
            c = self.activation(c)
            d = self.activation(d)
            o = self.activation(o)

            errorA = nn.functional.mse_loss(self.fcBA(b), a)
            reconstructionA = torch.autograd.grad(errorA, b, retain_graph=True)[0]

            errorB = nn.functional.mse_loss(self.fcCB(c), b)
            reconstructionB = torch.autograd.grad(errorB, c, retain_graph=True)[0]
            
            errorC = nn.functional.mse_loss(self.fcDC(d), c)
            reconstructionC = torch.autograd.grad(errorC, d, retain_graph=True)[0]

            aNew = gammaFw * self.fcin(i) + (1 - gammaFw - betaBw) * a + betaBw * self.fcBA(b)
            bNew = gammaFw * self.fcAB(aNew) + (1 - gammaFw - betaBw) * b + betaBw * self.fcCB(c) - self.alphaRec * batchSize * reconstructionA
            cNew = gammaFw * self.fcBC(bNew) +  (1 - gammaFw - betaBw) * c + betaBw * self.fcDC(d) - self.alphaRec * batchSize * reconstructionB
            dNew = gammaFw * self.fcCD(cNew) + (1 - gammaFw) * d - self.alphaRec * batchSize * reconstructionC
            oNew = self.fcout(dNew)

        out =  torch.log_softmax(oNew,dim=1)
        return out, i, aNew, bNew, cNew, dNew, oNew, reconstructionA, reconstructionB, reconstructionC

    