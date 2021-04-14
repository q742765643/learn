import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import numpy as np
import cv2
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import os
import json
import shutil
from paddle.fluid.incubate.fleet.parameter_server.distribute_transpiler import fleet
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid.transpiler.distribute_transpiler import DistributeTranspilerConfig
import sys
in_file_path=""
model_file_path=""
roles=1
EPOCH_NUM=1
server_endpoints=""
current_id=0
def createDataList(data_root_path,filePath):
        if(os.path.exists(filePath)):
           shutil.rmtree(filePath)
        # # 把生产的数据列表都放在自己的总类别文件夹中
        data_list_path = ''
        # 所有类别的信息
        class_detail = []
        # 获取所有类别
        class_dirs = os.listdir(data_root_path)
        # 类别标签
        class_label = 0
        # 获取总类别的名称
        father_paths = data_root_path.split('/')
        while True:
            if father_paths[father_paths.__len__() - 1] == '':
                del father_paths[father_paths.__len__() - 1]
            else:
                break
        father_path = father_paths[father_paths.__len__() - 1]

        all_class_images = 0
        # 读取每个类别
        for class_dir in class_dirs:
            # 每个类别的信息
            class_detail_list = {}
            test_sum = 0
            trainer_sum = 0
            # 把生产的数据列表都放在自己的总类别文件夹中
            data_list_path = filePath
            # 统计每个类别有多少张图片
            class_sum = 0
            # 获取类别路径
            path = data_root_path + "/" + class_dir
            # 获取所有图片
            img_paths = os.listdir(path)
            for img_path in img_paths:
                # 每张图片的路径
                name_path = path + '/' + img_path
                # 如果不存在这个文件夹,就创建
                isexist = os.path.exists(data_list_path)
                if not isexist:
                    os.makedirs(data_list_path)
                # 每10张图片取一个做测试数据
                if class_sum % 10 == 0:
                    test_sum += 1
                    with open(data_list_path + "test.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                else:
                    trainer_sum += 1
                    with open(data_list_path + "trainer.list", 'a') as f:
                        f.write(name_path + "\t%d" % class_label + "\n")
                class_sum += 1
                all_class_images += 1
            class_label += 1
            # 说明的json文件的class_detail数据
            class_detail_list['class_name'] = class_dir
            class_detail_list['class_label'] = class_label
            class_detail_list['class_test_images'] = test_sum
            class_detail_list['class_trainer_images'] = trainer_sum
            class_detail.append(class_detail_list)
        # 获取类别数量
        all_class_sum = class_dirs.__len__()
        # 说明的json文件信息
        readjson = {}
        readjson['all_class_name'] = father_path
        readjson['all_class_sum'] = all_class_sum
        readjson['all_class_images'] = all_class_images
        readjson['class_detail'] = class_detail
        jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
        with open(data_list_path + "readme.json",'w') as f:
            f.write(jsons)
        return all_class_sum

def dataMapper(sample):
    img, label = sample
    #解决中文路径问题
    #img = paddle.dataset.image.load_image(img)
    #img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
    #img = paddle.dataset.image.simple_transform(img, 32,32, True)
    #img = img.flatten().astype('float32')/255.0
    #return img, label
    #img = paddle.dataset.image.simple_transform(img, 32, 32, True)
    #return img.flatten().astype('float32')/255.0, label
    im = Image.open(img)
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
    return im,label

def dataReader(path):
    def reader():
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)
    buffered_size=1024
    return paddle.reader.xmap_readers(dataMapper, reader, cpu_count(), buffered_size)

#定义网络配置 VGG-13 进行网络搭建
def networkConfiguration(img,type_size):
    # 模块一：3*3*64双卷积+池化层 filter_size卷积核大小 num_filters积核数量，其与输入通道相同 act 激活函数
    conv1 = fluid.layers.conv2d(input=img,filter_size=3,num_filters=64,padding=1,act='relu')
    conv2 = fluid.layers.conv2d(input=conv1,filter_size=3,num_filters=64,padding=1,act='relu')
    pool1 = fluid.layers.pool2d(input=conv2,pool_size=2,pool_type='max',pool_stride=2)
    conv_pool_1 = fluid.layers.batch_norm(pool1)
    # 模块二：3*3*128双卷积+池化层
    conv3 = fluid.layers.conv2d(input=conv_pool_1,filter_size=3,num_filters=128,padding=1,act='relu')
    conv4 = fluid.layers.conv2d(input=conv3,filter_size=3,num_filters=128,padding=1,act='relu')
    pool2 = fluid.layers.pool2d(input=conv4,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
    conv_pool_2 = fluid.layers.batch_norm(pool2)
    # 模块三：3*3*256双卷积+池化层
    conv5 = fluid.layers.conv2d(input=conv_pool_2,filter_size=3,num_filters=256,padding=1,act='relu')
    conv6 = fluid.layers.conv2d(input=conv5,filter_size=3,num_filters=256,padding=1,act='relu')
    pool3 = fluid.layers.pool2d(input=conv6,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
    conv_pool_3 = fluid.layers.batch_norm(pool3)
    # 模块四：3*3*512双卷积+池化层
    conv7 = fluid.layers.conv2d(input=conv_pool_3,filter_size=3,num_filters=512,padding=1,act='relu')
    conv8 = fluid.layers.conv2d(input=conv7,filter_size=3,num_filters=512,padding=1,act='relu')
    pool4 = fluid.layers.pool2d(input=conv8,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
    conv_pool_4 = fluid.layers.batch_norm(pool4)
    # 模块五：3*3*512双卷积+池化层
    conv9 = fluid.layers.conv2d(input=conv_pool_4,filter_size=3,num_filters=512,padding=1,act='relu')
    conv10 = fluid.layers.conv2d(input=conv9,filter_size=3,num_filters=512,padding=1,act='relu')
    pool5 = fluid.layers.pool2d(input=conv10,pool_size=2,pool_type='max',pool_stride=2,global_pooling=False)
    # 以softmax 为激活函数的全连接输出层，10类数据输出10个数字
    fc1 = fluid.layers.fc(input=pool5,size=1000,act='relu')
    fc2 = fluid.layers.fc(input=fc1,size=1000,act='relu')
    prediction = fluid.layers.fc(input=fc2,size=type_size,act='softmax')
    return prediction

########## 模型训练＆模型评估 ##########
# 创建Executor
def save(predict,savaPath,exe):
    if not os.path.exists(savaPath):
     os.makedirs(savaPath)
    print('save models to %s' % (savaPath))
    fleet.save_inference_model(dirname=savaPath, feeded_var_names=['images'],target_vars=[predict], executor=exe)
    #fluid.io.save_inference_model(savaPath,['images'],[predict],exe)
def fit():
    role = role_maker.UserDefinedRoleMaker(
        current_id=current_id,
        role=role_maker.Role.WORKER if bool(1==int(roles)) else role_maker.Role.SERVER,
        worker_num=2,
        server_endpoints=["127.0.0.1:36011"])
    fleet.init(role)
    BATCH_SIZE = 128
    type_size=createDataList(model_file_path,model_file_path+'.data'+"/")
    # 用于训练的数据提供器
    train_reader=paddle.batch(reader=paddle.reader.shuffle(reader=dataReader(in_file_path+".data/trainer.list"),buf_size=BATCH_SIZE*100), batch_size=BATCH_SIZE)
    test_reader=paddle.batch(reader=paddle.reader.shuffle(reader=dataReader(in_file_path+".data/test.list"),buf_size=BATCH_SIZE*100), batch_size=BATCH_SIZE)
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    # 获取分类器
    predict = networkConfiguration(images,type_size)

    # 定义损失函数和准确率
    cost = fluid.layers.cross_entropy(input=predict, label=label)   # 交叉熵
    avg_cost = fluid.layers.mean(cost)                              # 计算cost中所有元素的平均值
    acc = fluid.layers.accuracy(input=predict, label=label)         # 使用输入和标签计算准确率

    # 定义优化方法
    test_program = fluid.default_main_program().clone(for_test=True)    # 获取测试程序
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    strategy = DistributeTranspilerConfig()
    strategy.sync_mode = True
    optimizer = fleet.distributed_optimizer(optimizer,strategy)
    # 定义优化方法
    optimizer.minimize(avg_cost)

    if fleet.is_server():
        print("启动server")
        fleet.init_server()
        fleet.run_server()

    elif fleet.is_worker():
        print("启动worker")
        ########## 模型训练＆模型评估 ##########
        # 创建Executor
        use_cuda = False # 定义使用CPU还是GPU，使用CPU时use_cuda=False
        place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
        print("cpu")
        # 定义数据映射器
        feeder = fluid.DataFeeder(feed_list=[images, label], place=place)
        print("数据映射")
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        fleet.init_worker()
        print(fleet.worker_endpoints())
        for pass_id in range(EPOCH_NUM):
            print(pass_id)
            # 开始训练
            for batch_id, data in enumerate(train_reader()):                            # 遍历train_reader
                train_cost, train_acc = exe.run(program=fluid.default_main_program(),   # 运行主程序
                                                feed=feeder.feed(data),                 # 喂入一个batch的数据
                                                fetch_list=[avg_cost, acc])             # fetch均方误差和准确率         # fetch均方误差和准确率
                # 每100次batch打印一次训练、进行一次测试
                if batch_id % 20 == 0:
                    print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %(pass_id, batch_id, train_cost[0], train_acc[0]))
            # 开始测试
            test_costs = [] # 测试的损失值
            test_accs = []  # 测试的准确率
            for batch_id, data in enumerate(test_reader()):
                test_cost, test_acc = exe.run(program=test_program,         # 执行训练程序
                                            feed=feeder.feed(data),       # 喂入数据
                                            fetch_list=[avg_cost, acc])   # fetch误差、准确率
                test_costs.append(test_cost[0])                             # 记录每个batch的损失值
                test_accs.append(test_acc[0])                               # 记录每个batch的准确率

            test_cost = (sum(test_costs) / len(test_costs)) # 计算误差平均值
            test_acc = (sum(test_accs) / len(test_accs))    # 计算准确率平均值
            print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
        save(predict,model_file_path,exe)
        fleet.stop_worker()

if __name__ == '__main__':
    args=sys.argv
    in_file_path=args[1]
    model_file_path=args[2]
    roles=args[3]
    EPOCH_NUM=int(args[4])
    server_endpoints=args[5]
    current_id=int(args[6])
    fit()





