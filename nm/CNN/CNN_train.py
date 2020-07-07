print(__doc__)

import os
import random
import sys

import cv2
import keras.backend as K
import matplotlib.pylab as plt
import numpy as np
from imutils import paths
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array  # 图片转为array
from keras.utils import to_categorical  # 相当于one-hot
from sklearn.model_selection import train_test_split


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
    dict['channel'] = int(args[2]);
    dict['batch_size'] = int(args[3]);
    dict['epochs'] = int(args[4]);
    dict['in_file_path'] = args[5];
    dict['out_file_path'] = args[6];
    return dict


def createDataSet(dict):
    path = dict['in_file_path']
    norm_size = dict['norm_size']
    channel = dict['channel']
    batch_size = dict['batch_size']
    epochs = dict['epochs']
    X = []
    Y = []
    image_paths = sorted(list(paths.list_images(path)))
    random.seed(0)  # 保证每次数据顺序一致
    random.shuffle(image_paths)  # 将所有的文件路径打乱
    for each_path in image_paths:
        image = cv2.imread(each_path)  # 读取文件
        # norm_size
        image = cv2.resize(image, (norm_size, norm_size))  # 统一图片尺寸
        image = img_to_array(image)
        X.append(image)
        label = each_path.split(os.path.sep)[-2]  # 切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        Y.append(label)

    X = np.array(X, dtype="float") / 255.0  # 归一化
    Y = np.array(Y)
    # 多标签二值化
    # mlb = MultiLabelBinarizer()
    # Y = mlb.fit_transform(Y)
    # print(mlb.classes_)
    Y = to_categorical(Y)
    num_classes = len(Y[0])
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")  # 数据增强，生成迭代器

    input_shape = (channel, norm_size, norm_size)
    if K.image_data_format() == "channels_last":  # 确认输入维度
        input_shape = (norm_size, norm_size, channel)
    model = Sequential()  # 顺序模型（keras中包括顺序模型和函数式API两种方式）
    model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=input_shape, name="conv1"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool1"))
    model.add(Conv2D(50, (5, 5), padding="same", activation="relu", name="conv2", ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="pool2"))
    model.add(Flatten())
    model.add(Dense(500, activation="relu", name="fc1"))
    model.add(Dense(num_classes, activation="softmax", name="fc2"))
    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam", metrics=["accuracy"])  # 配置

    (X, testX, Y, testY) = train_test_split(X, Y, test_size=0.2, random_state=42)
    # model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y))
    _history = model.fit_generator(aug.flow(X, Y, batch_size=batch_size),
                                   validation_data=(testX, testY), steps_per_epoch=len(X) // batch_size,
                                   epochs=epochs, verbose=1)
    model_path = dict['out_file_path']
    model_parent_path = os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    model.save(model_path)
    plt.style.use("ggplot")  # matplotlib的美化样式
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), _history.history["loss"],
             label="train_loss")  # model的history有四个属性，loss,val_loss,acc,val_acc
    plt.plot(np.arange(0, N), _history.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), _history.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), _history.history["val_accuracy"], label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    png_path = os.path.splitext(model_path)[0] + ".png"
    plt.savefig(png_path)
    # plt.show()


if __name__ == "__main__":
    main()
