import cv2
import numpy as np
from dataprocess.utils import file_name_path
import pandas as pd


def analyse_image(input_path):
    image = "image"
    file_image_dir = input_path + "/" + image

    file_paths = file_name_path(file_image_dir, dir=False, file=True)

    size = []
    for index in range(len(file_paths)):
        file_image_path = file_image_dir + "/" + file_paths[index]
        image = cv2.imread(file_image_path, 0)
        print(image.shape)
        size.append(np.array(image.shape))
    print("mean size:", (np.mean(np.array(size), axis=0)))


def splitclassifyintotraintest(file_name):
    classify = r'E:\MedicalData\TNSCUI2020\TNSCUI2020_train\train.csv'
    image_path = r"E:\MedicalData\TNSCUI2020\TNSCUI2020_train\image"
    csvdata = pd.read_csv(classify)
    traindata = csvdata.iloc[:, :].values
    imagedata = traindata[:, 0]
    labeldata = traindata[:, 1]
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for index in range(len(imagedata)):
        out.writelines(str(labeldata[index]) + "," + image_path + "/" + str(imagedata[index]) + "\n")


if __name__ == '__main__':
    # analyse_image(r'E:\MedicalData\TNSCUI2020\TNSCUI2020_train')
    splitclassifyintotraintest("classifydata.csv")
    print('success')
