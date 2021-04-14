import paddle.fluid as fluid
import paddle as paddle
import numpy as np
import cv2
import os
import json
from PIL import Image
import sys

model_path = ''
img_path = ''
def read_image(path):
    im = Image.open(path)
    # 将图片调整为训练数据同等大小
    # 设定ANTIALIAS，抗锯齿
    im = im.resize((32, 32), Image.ANTIALIAS)
    # 建立图片矩阵，类型为float32
    im = np.array(im).astype(np.float32)
    # 矩阵转置
    im = im.transpose((2, 0, 1))
    # 将像素值从0-255变成0-1
    im = im / 255.0
    im = np.expand_dims(im, axis=0)
    # 保持和之前输入image维度一致
    # print('im_shape的维度：', im.shape)
    return im
    # img = paddle.dataset.image.load_image(path)
    # #img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
    # #img = paddle.dataset.image.simple_transform(img, 32,32, True)
    # #img = img.flatten().astype('float32')/255.0
    # #return img, label
    # img = paddle.dataset.image.simple_transform(img, 32, 32, False)
    # return img.flatten().astype('float32')/255

if __name__ == '__main__':
    args=sys.argv
    model_path=args[1]
    img_path=args[2]

    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    [infer_program, feed_var_names, target_vars] = fluid.io.load_inference_model(dirname=model_path, executor=exe)
    print("==============")
    print(feed_var_names)
    print("==============")
    img = read_image(img_path)
    result = exe.run(program=infer_program, feed={feed_var_names[0] :img}, fetch_list=target_vars)
    max_id = np.argmax(result[0][0])
    cls_list = []
    with open(model_path+'.data/readme.json','r',encoding='utf8')as fp:
        json_data = json.load(fp)
        cls_list=json_data['class_detail']
    print(result[0][0])
    print('预测结果为：{:s}, 概率为：{:.3f}'.format(cls_list[max_id]['class_name'], result[0][0][max_id]))

