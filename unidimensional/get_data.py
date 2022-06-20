import numpy as np
import torch

def get_data(Nbsamples, ratio = (0.7,0.3)):

    train_size, test_size = ratio
    train_size, test_size = int(Nbsamples*train_size), int(Nbsamples*test_size)
    train = []
    test = []

    for _ in range(train_size):

        x = torch.rand(1)
        if x<0.5:
            y = 0
        else:
            y = 1
        train.append((x,y))
    
    for _ in range(test_size):

        x = torch.rand(1)
        if x<0.5:
            y = 0
        else:
            y = 1
        test.append((x,y))

    X,Y = [x for (x,_) in train], [y for (_,y) in train]

    train = Dataset(X,Y)

    X,Y = [x for (x,_) in test], [y for (_,y) in test]

    test = Dataset(X,Y)    

    return train,test


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_data, y_labels):
        'Initialization'
        self.y = y_labels
        self.x = x_data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        X = self.x[index].unsqueeze_(0)
        y = self.y[index]

        return X, y

