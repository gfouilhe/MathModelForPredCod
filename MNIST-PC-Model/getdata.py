from turtle import color
import torch
import numpy
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def get_data():
    train = datasets.MNIST(os.path.join('data'),train=True,download=False,transform=transforms.ToTensor())
    test = datasets.MNIST(os.path.join('data'),train=False,download=False,transform=transforms.ToTensor())
    return train,test

train,test = get_data()

#print(train)
#print(test)


def plot_data(data):
    plt.figure()
    for i in range(9):
        image,_ = data[i]
        print(image)
        plt.subplot(3,3,i+1)
        plt.imshow(image.reshape((28,28)),cmap='gray')
    plt.show()

plot_data(train)

