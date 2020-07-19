import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Vnet2d.vnet_seg_classify_model import Vnet2dmutiltaskModule
from dataprocess.utils import calcu_iou
import cv2
import os
import numpy as np


def predict_test():
    name = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\seg_classifyvnetmask.csv"
    out = open(name, 'w')
    out.writelines("ID" + "," + "CATE" + "\n")
    Vnet2d = Vnet2dmutiltaskModule(512, 512, channels=1, n_class=2, costname=("dice coefficient", "cross_entropy"),
                                   inference=True, model_path="log\segclasif\\vnet2d\model\Vnet2d.pd")
    test_image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\image"
    test_mask_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\seg_classifyvnetmask"
    for number in range(1, 911, 1):
        imagefile = "test_" + str(number) + ".PNG"
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        pd_mask_image, label_image = Vnet2d.prediction(resize_image / 255.)
        new_mask_image = cv2.resize(pd_mask_image, (src_image.shape[1], src_image.shape[0]))
        maskfilepath = os.path.join(test_mask_path, imagefile)
        cv2.imwrite(maskfilepath, new_mask_image)
        out.writelines(imagefile + "," + str(label_image[0]) + "\n")


if __name__ == "__main__":
    predict_test()
    print('success')
