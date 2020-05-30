import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv(r'D:\cjq\project\python\TNSCUI2020\dataprocess\segmentationtraindata.csv')
    csvdataaug = pd.read_csv(r'D:\cjq\project\python\TNSCUI2020\dataprocess\segmeatationaugtraindata.csv')
    traindata = csvdata.iloc[:, :].values
    augtraindata = csvdataaug.iloc[:, :].values
    trainData = np.concatenate((traindata, augtraindata), axis=0)
    # shuffle imagedata and maskdata together
    np.random.shuffle(trainData)
    imagedata = trainData[:, 0]
    maskdata = trainData[:, 1]

    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient")
    Vnet2d.train(imagedata, maskdata, "Vnet2d.pd", "log\\segmeation\\vnet2d\\", 0.001, 0.5, 5, 6)


if __name__ == '__main__':
    train()
    print('success')
