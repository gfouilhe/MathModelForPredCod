import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import PCMLP
from PIL import Image
import pickle

rgba2gray = lambda x: np.dot(x[...,:3], [0.2989, 0.5870, 0.1140])

for alpha in [0.01,0.05,0.1,0.25]:
    with open(os.path.join('oscillations_parameters_setup',f'params_dictionary_{alpha}.pkl'), 'rb') as f:
        params_and_imgs = pickle.load(f)
    params_list = [param for _,param in params_and_imgs.items()]
    l = len(params_list)
    imgs = [rgba2gray(plt.imread(os.path.join('oscillations_parameters_setup',f'img_{alpha}_{i}.png'))) for i in range(l)]
    M = imgs[0].shape[0] #14
    mean_img = np.zeros((M,M))
    for m in range(M):
        for n in range(M):
            mean_img[m,n] = 1/l*sum([img[m,n] for img in imgs])
    plt.imsave(os.path.join('oscillations_parameters_setup',f'mean_img_{alpha}.png'),mean_img,cmap='gray')