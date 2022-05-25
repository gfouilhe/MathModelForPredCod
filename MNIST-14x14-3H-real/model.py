import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PCMLP(nn.Module):
    '''
    Architechture :
    all FC

    196 --> 196 --> 196 --> 10 
    I        A       B       O
    |---R <--|--R <--|

    
    '''
    def __init__(self, memory, alphaRec, betaFB, gammaFw,linear=False,transp=False):
        
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
        #self.fcBC = nn.Linear(self.num_hidden,self.num_hidden)
        #self.fcCB = nn.Linear(self.num_hidden,self.num_hidden)
        self.fcout = nn.Linear(self.num_hidden,10)
        self.activation = F.relu
        if linear:
            self.activation = lambda x: x
        self.transp = transp


    def forward(self, i, a, b, o, networkMode):

        errorI=[]
        reconstructionI = []

        errorA = []
        reconstructionA = []

        batchSize = a.shape[0]


        assert networkMode in ['forward','full']

        if networkMode == 'forward':

            aNew = self.activation(self.fciA(i))
            bNew= self.activation(self.fcAB(aNew))
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

            if self.transp:
                errorI = nn.functional.mse_loss(torch.matmul(a, self.fciA.weight), i)
                reconstructionI = torch.autograd.grad(errorI, a, retain_graph=True)[0]

                errorA = nn.functional.mse_loss(torch.matmul(b, self.fcAB.weight), a)
                reconstructionA = torch.autograd.grad(errorA, b, retain_graph=True)[0]


                aNew = gammaFw * self.activation(self.fciA(i)) + (1 - gammaFw - betaBw) * a + betaBw * self.activation(torch.matmul(b, self.fcAB.weight)) - self.alphaRec * batchSize * reconstructionI
                bNew = gammaFw * self.activation(self.fcAB(aNew)) + + (1 - gammaFw) * b - self.alphaRec * batchSize * reconstructionA
                oNew = self.fcout(bNew)
            else:
                errorI = nn.functional.mse_loss(self.fcAi(a), i)
                reconstructionI = torch.autograd.grad(errorI, a, retain_graph=True)[0]

                errorA = nn.functional.mse_loss(self.fcBA(b), a)
                reconstructionA = torch.autograd.grad(errorA, b, retain_graph=True)[0]


                aNew = gammaFw * self.activation(self.fciA(i)) + (1 - gammaFw - betaBw) * a + betaBw * self.activation(self.fcBA(b)) - self.alphaRec * batchSize * reconstructionI
                bNew = gammaFw * self.activation(self.fcAB(aNew)) + + (1 - gammaFw) * b - self.alphaRec * batchSize * reconstructionA
                oNew = self.fcout(bNew)

        out =  torch.log_softmax(oNew,dim=1)
        return out, i, aNew, bNew,  oNew, reconstructionA

    