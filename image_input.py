from asyncore import read
from cgi import test
import torch
from configparser import Interpolation
import cv2
import matplotlib.pyplot as plt

def get_img(filename):

    test_image = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
    plt.imshow(test_image,cmap = "gray")
    img_resized = cv2.resize(test_image,(28,28),interpolation = cv2.INTER_CUBIC)
    img_resized = cv2.bitwise_not(img_resized)
    img_resized =  torch.from_numpy(img_resized)  
    img_resized = img_resized.type(torch.float32)
    img_resized/=256
    return img_resized