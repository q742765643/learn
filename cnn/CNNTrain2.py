import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import numpy as np
import cv2
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import os
def __init__(self,imageSize):
    self.imageSize = imageSize
########## 准备数据 ##########
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def test_mapper(self,sample):
    img, label = sample
    # 将img 数组进行归一化处理，得到0到1之间的数值
    img = paddle.image.load_image(img)
    img = paddle.dataset.image.simple_transform(img, 70, self.imageSize, True)
    img = img.flatten().astype('float32')/255.0
    return img, label

def train_mapper(sample):
    img, label = sample
    img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), 1)
    print(len(img),len(img[0]),len(img[0][0]))#打印维度信息
    #img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(img, 32,32, True)
    img = img.flatten().astype('float32')/255.0
    return img, label

# 对自定义数据集创建训练集train 的reader
def train_r(buffered_size=1024):
    def reader():
        with open("D:/迅雷下载/cifar-10-python/data/trainer.list", 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)
    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), buffered_size)

# 对自定义数据集创建训练集test 的reader
def test_r(buffered_size=1024):
    def reader():
        test_dict = unpickle("D:\迅雷下载\cifar-10-batches-py/test_batch")
        X = test_dict[b'data']
        Y = test_dict[b'labels']
        for (x, y) in zip(X, Y):
            yield x, int(y)
    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), buffered_size)

BATCH_SIZE = 128
# 用于训练的数据提供器
train_reader = train_r()
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader = train_reader, buf_size = BATCH_SIZE*100),
    batch_size = BATCH_SIZE
    )
# 用于测试的数据提供器
test_reader = test_r()
test_reader = paddle.batch(
    paddle.dataset.cifar.test10(),
    batch_size = BATCH_SIZE
    )

########## 网络配置 ##########
def convolutional_neural_network(img):
    # 模块一：3*3*64双卷积+池化层
    conv1 = fluid.layers.conv2d(input=img,          # 输入图像
                                filter_size=3,      # 卷积核大小
                                num_filters=64,     # 卷积核数量，其与输入通道相同
                                padding=1,
                                act='relu')         # 激活函数

    conv2 = fluid.layers.conv2d(input=conv1,          # 输入图像
                                filter_size=3,      # 卷积核大小
                                num_filters=64,     # 卷积核数量，其与输入通道相同
                                padding=1,
                                act='relu')         # 激活函数

    pool1 = fluid.layers.pool2d(input=conv2,        # 输入
                                pool_size=2,        # 池化核大小
                                pool_type='max',    # 池化类型
                                pool_stride=2)      # 池化步长

    conv_pool_1 = fluid.layers.batch_norm(pool1)

    # 模块二：3*3*128双卷积+池化层
    conv3 = fluid.layers.conv2d(input=conv_pool_1,
                                filter_size=3,
                                num_filters=128,
                                padding=1,
                                act='relu')

    conv4 = fluid.layers.conv2d(input=conv3,
                                filter_size=3,
                                num_filters=128,
                                padding=1,
                                act='relu')

    pool2 = fluid.layers.pool2d(input=conv4,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2,
                                global_pooling=False)

    conv_pool_2 = fluid.layers.batch_norm(pool2)

    # 模块三：3*3*256双卷积+池化层
    conv5 = fluid.layers.conv2d(input=conv_pool_2,
                                filter_size=3,
                                num_filters=256,
                                padding=1,
                                act='relu')

    conv6 = fluid.layers.conv2d(input=conv5,
                                filter_size=3,
                                num_filters=256,
                                padding=1,
                                act='relu')

    pool3 = fluid.layers.pool2d(input=conv6,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2,
                                global_pooling=False)

    conv_pool_3 = fluid.layers.batch_norm(pool3)

    # 模块四：3*3*512双卷积+池化层
    conv7 = fluid.layers.conv2d(input=conv_pool_3,
                                filter_size=3,
                                num_filters=512,
                                padding=1,
                                act='relu')

    conv8 = fluid.layers.conv2d(input=conv7,
                                filter_size=3,
                                num_filters=512,
                                padding=1,
                                act='relu')

    pool4 = fluid.layers.pool2d(input=conv8,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2,
                                global_pooling=False)

    conv_pool_4 = fluid.layers.batch_norm(pool4)

    # 模块五：3*3*512双卷积+池化层
    conv9 = fluid.layers.conv2d(input=conv_pool_4,
                                filter_size=3,
                                num_filters=512,
                                padding=1,
                                act='relu')

    conv10 = fluid.layers.conv2d(input=conv9,
                                filter_size=3,
                                num_filters=512,
                                padding=1,
                                act='relu')

    pool5 = fluid.layers.pool2d(input=conv10,
                                pool_size=2,
                                pool_type='max',
                                pool_stride=2,
                                global_pooling=False)

    # 以softmax 为激活函数的全连接输出层，10类数据输出10个数字
    fc1 = fluid.layers.fc(input=pool5,
                                 size=1000,
                                 act='relu')
    fc2 = fluid.layers.fc(input=fc1,
                                 size=1000,
                                 act='relu')
    prediction = fluid.layers.fc(input=fc2,
                                 size=10,
                                 act='softmax')
    return prediction

# 定义输入数据
data_shape = [3, 32, 32]
paddle.enable_static()
images = fluid.layers.data(name='images', shape=data_shape, dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取分类器
predict = convolutional_neural_network(images)

# 定义损失函数和准确率
cost = fluid.layers.cross_entropy(input=predict, label=label)   # 交叉熵
avg_cost = fluid.layers.mean(cost)                              # 计算cost中所有元素的平均值
acc = fluid.layers.accuracy(input=predict, label=label)         # 使用输入和标签计算准确率

# 定义优化方法
test_program = fluid.default_main_program().clone(for_test=True)    # 获取测试程序
optimizer = fluid.optimizer.Adam(learning_rate=0.001)               # 定义优化方法
optimizer.minimize(avg_cost)
print("完成")

########## 模型训练＆模型评估 ##########
# 创建Executor
use_cuda = False # 定义使用CPU还是GPU，使用CPU时use_cuda=False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# 定义数据映射器
feeder = fluid.DataFeeder(feed_list=[images, label], place=place)

# 定义绘制训练过程的损失值和准确率变化趋势的方法
all_train_iter = 0
all_train_iters = []
all_train_costs = []
all_train_accs = []

def draw_train_process(title, iters, costs, accs, label_cost, label_acc):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs, color='red', label=label_cost)
    plt.plot(iters, accs, color='green', label=label_acc)
    plt.legend()
    plt.grid()
    plt.show()

# 训练并保存模型
EPOCH_NUM = 3
model_save_dir = "/home/aistudio/model/DogCat_Detection.inference.model"

for pass_id in range(EPOCH_NUM):
    # 开始训练
    for batch_id, data in enumerate(train_reader()):                            # 遍历train_reader
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),   # 运行主程序
                                        feed=feeder.feed(data),                 # 喂入一个batch的数据
                                        fetch_list=[avg_cost, acc])             # fetch均方误差和准确率

        all_train_iter = all_train_iter + BATCH_SIZE
        all_train_iters.append(all_train_iter)
        all_train_costs.append(train_cost[0])
        all_train_accs.append(train_acc[0])

        # 每100次batch打印一次训练、进行一次测试
        if batch_id % 20 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy:%0.5f' %
                  (pass_id, batch_id, train_cost[0], train_acc[0]))

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

# 保存模型
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)
print('save models to %s' % (model_save_dir))
fluid.io.save_inference_model(model_save_dir,
                              ['images'],
                              [predict],
                              exe)

print('模型保存完成')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
draw_train_process("training", all_train_iters, all_train_costs, all_train_accs, "training cost", "training acc")
