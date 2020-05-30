import os
import numpy as np


def calcu_dice(Y_pred, Y_gt, K=255):
    """
    calculate two input dice value
    :param Y_pred:
    :param Y_gt:
    :param K:
    :return:
    """
    intersection = 2 * np.sum(Y_pred[Y_gt == K])
    denominator = np.sum(Y_pred) + np.sum(Y_gt)
    loss = (intersection / denominator)
    return loss


def calcu_iou(Y_pred, Y_gt, K=255):
    """
    calculate two input iou value
    :param Y_pred:
    :param Y_gt:
    :param K:
    :return:
    """
    intersection = np.sum(Y_pred[Y_gt == K])
    denominator = np.sum(Y_pred) + np.sum(Y_gt) - intersection
    loss = (intersection / denominator)
    return loss


def file_name_path(file_dir, dir=True, file=False):
    """
    get root path,sub_dirs,all_sub_files
    :param file_dir:
    :return:
    """
    for root, dirs, files in os.walk(file_dir):
        if len(dirs) and dir:
            print("sub_dirs:", dirs)
            return dirs
        if len(files) and file:
            print("files:", files)
            return files


def save_file2csv(file_dir, file_name):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    image = "image"
    mask = "mask"
    file_image_dir = file_dir + "/" + image
    file_mask_dir = file_dir + "/" + mask
    file_paths = file_name_path(file_image_dir, dir=False, file=True)
    out.writelines("Image,Mask" + "\n")
    for index in range(len(file_paths)):
        out_file_image_path = file_image_dir + "/" + file_paths[index]
        out_file_mask_path = file_mask_dir + "/" + file_paths[index]
        out.writelines(out_file_image_path + "," + out_file_mask_path + "\n")


def save_file2csvclassify(file_dir, file_name, labelnum=2):
    """
    save file path to csv
    :param file_dir:preprocess data path
    :param file_name:output csv name
    :return:
    """
    out = open(file_name, 'w')
    out.writelines("label,Image" + "\n")
    for i in range(labelnum):
        file_image_dir = file_dir + "/" + str(i)
        file_paths = file_name_path(file_image_dir, dir=False, file=True)
        for index in range(len(file_paths)):
            out_file_image_path = file_image_dir + "/" + file_paths[index]
            out.writelines(str(i) + "," + out_file_image_path + "\n")


if __name__ == '__main__':
    # save_file2csv(r"E:\MedicalData\TNSCUI2020\segmentation\augtrain", "segmeatationaugtraindata.csv")
    save_file2csvclassify(r"E:\MedicalData\TNSCUI2020\classifiy\augtrain", "classifyaugtraindata.csv")
    print('success')
