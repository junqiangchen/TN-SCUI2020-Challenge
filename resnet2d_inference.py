import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from Resnet2d.model_resNet2d import ResNet2dModule
import cv2


def predict():
    ResNet3d = ResNet2dModule(512, 512, channels=1, n_class=2, costname="cross_entropy", inference=True,
                              model_path=r"log\classify\resnetcross_entropy\model\resnet.pd")
    test_image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\image"
    name = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_test\classification.csv"
    out = open(name, 'w')
    out.writelines("ID" + "," + "CATE" + "\n")
    for number in range(1, 911, 1):
        imagefile = "test_" + str(number) + ".PNG"
        imagefilepath = os.path.join(test_image_path, imagefile)
        src_image = cv2.imread(imagefilepath, cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (512, 512))
        predictvalue, predict_prob = ResNet3d.prediction(resize_image / 255.)
        out.writelines(imagefile + "," + str(int(predictvalue[0])) + "\n")


if __name__ == "__main__":
    predict()
