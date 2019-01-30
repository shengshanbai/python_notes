import collections
import os
import cv2
import math
import matplotlib.pyplot as plt
import numpy

#这个文件用于从原始的imagenet数据中产生出训练需要的tensorflow database
IMAGE_NET_DIR='/home/ssbai/data/ImageNet'
ImageItem=collections.namedtuple('ImageItem','path pointStart size hflip pca pcaArgs tag')
#输入图片的标准大小
IMAGE_IN_SIZE=(256,256)

#得到所有的训练数据
def allTrainImages():
    images=list()
    trainFile=os.path.join(IMAGE_NET_DIR,'train.txt')
    with open(trainFile,'r') as trainContent:
        for line in trainContent.readlines():
            path,tag=line.split()
            fullpath=os.path.join(IMAGE_NET_DIR,'ILSVRC2012_img_train',path)
            image=cv2.imread(fullpath)
            image=resizeImage(image)

#输入图片需要缩放到网络需要的256*256大小
def resizeImage(image):
    rows,cols,_=image.shape
    if rows > cols:
        image=image[(rows-cols)//2:(rows+cols)//2,:]
    else:
        image=image[:,(cols-rows)//2:(cols+rows)//2]
    resized=cv2.resize(image,dsize=IMAGE_IN_SIZE)
    return resized

def main():
    allTrainImages()

if __name__=='__main__':
    main()