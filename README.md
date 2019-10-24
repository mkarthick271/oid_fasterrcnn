# Pytorch Implementation of Faster R-CNN for Google open images dataset 2019

## Introduction

  I have adapted the pytorch implementation of Faster R-CNN which can be found [here](https://github.com/jwyang/faster-rcnn.pytorch) to use Google open images dataset 2019(OID). This image dataset for object detection is the largest available dataset right now, which has over 1.7million training images to train from. It has over 600 boxable object classes which are also organized as a class hierarchy which makes things more complicated as the model should output a bounding box for a class and also output bounding boxes for it's parent classes. 
  
 The image labels and bounding boxes and other information about the images are given as .csv files. To analyze the huge amount of information given in the .csv files for this dataset, I have used PostgreSQL to which the .csv files are loaded to tables. Now, the dataset can be analyzed by quering with SQL statements, like to know the number of images per object class, the images which contains the largest number of bounding boxes and images which contain the largest number of unique object class labels etc. 
 
 Once the data is loaded to the tables in PostgreSQL, I have randomly sampled 250 images per object class and used these images to train the Faster R-CNN network. It took around 3 days to fully train the model for 20 epochs for 112,000 images on 2 nVidia V100 GPUS with a batch size of 8(4 images on each GPU) on Google cloud platform. Next I validated the model on 41,620 validation set of OID and got an **mAP of 39%** which is the baseline for this model. No image augmentations were used and plain Faster R-CNN network was used. This is a work in progress, and by using image augmentations and by using more latest networks the mAP is expected to improve. 


## Installation/Data Preparation(Please follow the below steps exactly in same sequence to train the network successfully) 
1. clone the code to your home directory - 
  git clone https://github.com/mkarthick271/oid_fasterrcnn.git
2. Install PostgreSQL and set the password of user postgres to cts-0000

3. pip install -r requirements.txt --user

4. pip install psycopg2-binary --user

5. cd into the lib directory and install this code  by running the make command  - $ make

6. cd into data/train and run the down.sh script to download the necessary oid .csv files for training

7. Add heading as labelname,labeldesc to the file challenge-2019-classes-description-500.csv

8. Run delcol.py program to delete some of the unwanted columns from train-images-boxable-with-rotation.csv file

9. Change $HOME$ in the file fasterrcnn.sql to your home directory

10. Run command 'psql -U postgres -h 127.0.0.1 -a -f fasterrcnn.sql' command to create the necessary tables and load the .csv data to the tables in PostgreSQL. Enter password as cts-0000 if prompted

11. Run the program dbcon.py to randomly sample a set of 250 images for each of the 500 object classes. 

12. Run command 'psql -U postgres -h 127.0.0.1 -a -f dataset250.sql' command to create table for the ground truth bounding boxes for the sampled training images and load the data. Enter password as cts-0000 if prompted.

13. Create a folder named 'dataset250' and run the program get_train250set.py to download the sampled images from the aws s3 bucket. This might take a long time or less, depending upon the number of  images you are downloading from aws.

14. Copy the pretrained backbone for resnet from [here](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0) and keep it in data/pretrained_model folder and make sure to change the name of the downloaded pretrained model to resnet101_caffe.pth

15. cd to the root oid_fasterrcnn directory and run the below command to train the model in background. 
      nohup bash -c "python -u trainval_net.py --cuda --net res101 --bs 4" &
      
      Here --bs is the batch size which is 4 assuming there is only one GPU in your system. Up the batch size by multiples of 4 as you add  more GPUs to your system. For example, if you have 8 GPUs use batch size of 32. If the code fails due to low memory on GPU, reduce the batch size on each GPU to 3 where --bs will be 24 for 8 GPUs. 
      
16. The trained model will be available in the path ./models/res101/oid/

## Steps to validate the trained model on the validation set of OID:

1. create folder ./data/valoiddata/validation

2. Download the file validation-images-with-rotation.csv from [here](https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv) and place it in the folder ./data/valoiddata

3. Copy the entire validation set of OID from aws s3 bucket which is [here](https://s3.console.aws.amazon.com/s3/buckets/open-images-dataset/validation/?region=ap-south-1) and place it in ./data/valoiddata/validation

4. Download the trained Faster R-CNN model from [here](https://storage.cloud.google.com/oidtrainedmodel/faster_rcnn_1_20_13908.pth?authuser=1) and place it in the folder ./models/res101/oid

5. cd to the root oid_fasterrcnn directory and run the below command to validate the model in background against the OID validation set. 
      nohup bash -c "python -u testoid_net.py --cuda --net res101" &
6. After the validation is completed, valsetpreds.csv file will be created which contains the bounding box detections of our model. In order to evaluate the mAP for this detection use valsetpreds.csv file and follow the steps given [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md#object-detection-track). This tutorial will help you to evaluate the mAP for each class of the OID.


