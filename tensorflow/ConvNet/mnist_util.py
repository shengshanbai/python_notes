# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:01:32 2018

@author: admin
"""
import gzip
import struct
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import cv2

IMAGE_SIZE=28

class MnistReader:
    def __init__(self,images_path,labels_path):
        self.images_path=images_path
        self.labels_path=labels_path
        self._open_images_labels()
        
    def _open_images_labels(self):
        self.image_file=gzip.GzipFile(self.images_path)
        magic_number=self._read_int(self.image_file)
        self.image_numbers=self._read_int(self.image_file)
        self.rows=self._read_int(self.image_file)
        self.columns=self._read_int(self.image_file)
        print("file:{} has {} images,image row {} image columns {}".format(self.images_path,
              self.image_numbers,self.rows,self.columns))
        self.labels_file=gzip.GzipFile(self.labels_path)
        magic_number=self._read_int(self.labels_file)
        self.labels_number=self._read_int(self.labels_file)
        print("file:{} has {} labels".format(self.labels_path,self.labels_number))
        if self.labels_number!=self.image_numbers:
            raise ValueError("the lables number not equals to image number!")
            
    def _read_int(self,file):
        int_read=file.read(4)
        return struct.unpack('!i',int_read)[0]
    
    def _read_byte(self,file):
        int_read=file.read(1)
        return struct.unpack('!B',int_read)[0]
        
    def _read_image(self,file):
        data_read=file.read(self.rows*self.columns)
        image=np.frombuffer(data_read,np.uint8).reshape((self.rows,self.columns,1))
        return image
        
    def _read_lable_(self,file):
        return self._read_byte(file)
        
    def gen_data(self):
        for i in range(self.image_numbers):
            image=self._read_image(self.image_file)
            image=image*(1/255)
            label=self._read_lable_(self.labels_file)
            one_hot=np.eye(10)[:,label]
            yield (image,one_hot)
    
    def close(self):
        if not self.image_file.closed():
            self.image_file.close()

def batch_gen(batch_size,image_path,label_path):
    mnistReader=MnistReader(image_path,label_path)
    gen=mnistReader.gen_data()
    while True:
        images=np.zeros((batch_size,mnistReader.rows,mnistReader.columns,1))
        labels=np.zeros((batch_size,10))
        for index in range(batch_size):
            images[index,:],labels[index,:]=next(gen)
        yield images,labels
    mnistReader.close()
        