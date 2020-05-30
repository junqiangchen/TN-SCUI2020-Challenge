import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
import cv2
import pandas as pd
import numpy as np


def predict():
    test_datacsv = pd.read_csv(r'D:\cjq\project\python\TNSCUI2020\dataprocess\classifytestdata.csv')
    test_data = test_datacsv.iloc[:, :].values
    # For Image
    images = test_data[:, 1]
    # For Labels
    labels = test_data[:, 0]
    ResNet3d = ResNet2dModule(512, 512, channels=1, n_class=2, costname="cross_entropy", inference=True,
                              model_path=r"log\classify\resnet\model\resnet.pd-100000")

    predictvalues = []
    predict_probs = []
    for num in range(len(images)):
        src_image = cv2.imread(images[num], 0)
        resize_image = cv2.resize(src_image, (512, 512))
        predictvalue, predict_prob = ResNet3d.prediction(resize_image / 255.)
        predictvalues.append(predictvalue)
        predict_probs.append(predict_prob)

    name = 'classify_metrics.csv'
    out = open(name, 'w')
    out.writelines("y_predict" + "," + "y_score" + "," + "y_true" + "\n")
    labels = labels.tolist()
    for index in range(np.shape(images)[0]):
        out.writelines(
            str(predictvalues[index][0]) + "," + str(predict_probs[index][0]) + "," + str(labels[index]) + "\n")


if __name__ == "__main__":
    predict()
