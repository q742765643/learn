print(__doc__)

import os
import sys

import cv2
import keras
import numpy as np
import pandas as pd
from imutils import paths
from keras.preprocessing.image import img_to_array  # 图片转为array


def main():
    print('参数个数为:', len(sys.argv), '个参数。')
    print('参数列表:', str(sys.argv))
    print('脚本名为：', sys.argv[0])
    for i in range(1, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))
    dict = createDict(sys.argv)
    createDataSet(dict)


# 创建参数生成字典
def createDict(args):
    dict = {};
    dict['norm_size'] = int(args[1]);
    dict['model_file_path'] = args[2];
    dict['in_file_path'] = args[3];
    dict['out_file_path'] = args[4];
    return dict


def createDataSet(dict):
    norm_size = dict['norm_size']
    image_paths = sorted(list(paths.list_images(dict['in_file_path'])))  # 提取图片
    model = keras.models.load_model(dict['model_file_path'])  # 加载模型
    data_list = []
    for each in image_paths:
        image = cv2.imread(each)
        image = cv2.resize(image, (norm_size, norm_size))  # 读取图片，修改尺寸
        image = img_to_array(image) / 255.0  # 归一化
        image = np.expand_dims(image, axis=0)  # 单张图片，改变维度
        result = model.predict(image)  # 分类预测
        proba = np.max(result)  # 最大概率
        predict_label = np.argmax(result)  # 提取最大概率下标
        data = []
        data.append(os.path.abspath(each))
        data.append(predict_label)
        data.append(proba)
        # print(data)
        data_list.append(data)

    out_path = dict['out_file_path']
    out_parent_path = os.path.split(out_path)[0]
    if not os.path.exists(out_parent_path):
        os.makedirs(out_parent_path)

    column = ['path', 'predict', 'predict_proba']
    csvList = pd.DataFrame(columns=column, data=data_list)
    csvList.to_csv(out_path, index=0)


if __name__ == "__main__":
    main()
