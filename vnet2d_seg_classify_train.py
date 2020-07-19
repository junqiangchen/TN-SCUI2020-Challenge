import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_seg_classify_model import Vnet2dmutiltaskModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\segandclassifydata.csv')
    trainData = csvdata.iloc[:, :].values
    np.random.shuffle(trainData)
    labeldata = trainData[:, 0]
    imagedata = trainData[:, 1]
    maskdata = trainData[:, 2]

    Vnet2d = Vnet2dmutiltaskModule(512, 512, channels=1, n_class=2, costname=("dice coefficient", "cross_entropy"))
    Vnet2d.train(imagedata, maskdata, labeldata, "Vnet2d.pd", "log\\segclasif\\vnet2d\\", 0.001, 0.5, 30, 6)


if __name__ == '__main__':
    train()
    print('success')
