from turtle import color
import torch
import numpy
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_data(n=100,std=None,scale_factor=0.85):

    X,Y = make_circles(n_samples=n,noise=std,factor=scale_factor)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def plot_data(X,Y,max_n = 200):
    n = X.shape[0]
    for i in range(max_n):
        if i < n :
            x,y = X[i]
            if Y[i] == 1:
                plt.scatter(x,y,color='red')
            else:
                plt.scatter(x,y,color='blue')
    plt.show()


#X_train, X_test, y_train, y_test = get_data(5000,0.1,scale_factor=0.85)
#plot_data(X_train,y_train)



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


