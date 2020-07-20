# TN-SCUI2020-Challenge
> This is an example of the Us imaging is used to segment and classify thyroid nodule.
![](tnscui2.png)

## Prerequisities
The following dependencies are needed:
- numpy >= 1.11.1
- SimpleITK >=1.0.1
- tensorflow-gpu ==1.14.0
- pandas >=0.20.1
- scikit-learn >= 0.17.1

## How to Use
* 1、when download the all project,check out the segmeatationdata.csv and classifydata.csv,put your train data into same folder.
* 2、run vnet2d_train.py for vnet2d segmeatation training:make sure train data have effective path
* 3、run vnet2d_inference.py for vnet2d segmeatation inference:make sure test data have effective path
* 4、run resnet2d_train.py for resnet2d classify training:make sure train data have effective path
* 5、run resnet2d_inference.py for resnet2d classify inference:make sure test data have effective path

## Result

* segment train loss and train accuracy
![](loss.PNG)

* classify train loss and train accuracy
![](loss2.PNG)

* test dataset segmentation result：left is source image,median is ground truth mask,right is predict mask
![](分割结果.png)

* test dataset leadboard
![](leadboard.png)

* more detail and trained model can follow my WeChat Public article.

## Contact
* https://github.com/junqiangchen
* email: 1207173174@qq.com
* Contact: junqiangChen
* WeChat Number: 1207173174
* WeChat Public number: 最新医学影像技术
