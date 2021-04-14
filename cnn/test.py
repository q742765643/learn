
#导入必要的包
import os
import paddle.dataset.image as II
import numpy as np
from PIL import Image
import paddle.fluid as fluid
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
def data_mapper(sample):
   img_path, label = sample
   #进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换
   img = II.load_image(img_path)
   img = paddle.dataset.image.simple_transform(im=img,
                                               resize_size=47,
                                               crop_size=47,
                                               is_color=True,
                                               is_train=True)
   img= img.flatten().astype('float32')/255.0
   return img, label


def data_r(file_list,buffered_size=1024):
   def reader():
       with open(file_list, 'r') as f:
            lines = [line.strip() for line inf]
            for line in lines:
                img_path, lab =line.strip().split('\t')
                yield img_path, int(lab)
   return paddle.reader.xmap_readers(data_mapper, reader,cpu_count(),buffered_size)
