import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_model import Vnet2dModule, AGVnet2dModule
from dataprocess.utils import calcu_iou
import cv2
import os
import numpy as np


def predict_test():
    Vnet2d = Vnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                          model_path="log\segmeation\\vnet2d\model\Vnet2d.pd")
    test_image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\image"
    test_mask_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\vnet_mask"
    allimagefiles = os.listdir(test_image_path)
    for imagefile in allimagefiles:
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)


def predict_testag():
    Vnet2d = AGVnet2dModule(512, 512, channels=1, costname="dice coefficient", inference=True,
                            model_path="log\segmeation\\agvnet2d\model\\agVnet2d.pd")
    test_image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\image"
    test_mask_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\agvnet_mask"
    allimagefiles = os.listdir(test_image_path)
    for imagefile in allimagefiles:
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)


if __name__ == "__main__":
    predict_test()
    # predict_testag()
    print('success')
