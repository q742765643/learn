print(__doc__)

import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.manifold import TSNE

def main():

    print('参数个数为:', len(sys.argv), '个参数。')
    print('参数列表:', str(sys.argv))
    print('脚本名为：', sys.argv[0])
    for i in range(1, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))
    createDataSet()
# 创建参数生成字典
def createDict(args):
    dict={};
    dict['C']=float(args[1]);
    dict['kernel']=args[2];
    dict['degree']=int(args[3]);
    dict['gamma']=args[4];
    dict['coef0']=float(args[5])
    dict['shrinking']=bool(args[6]);
    dict['probability']=bool(args[7]);
    dict['tol']=float(args[8]);
    dict['cache_size']=int(args[9]);
    if args[10]=='None':
       args[10]=None
    dict['class_weight']=args[10];
    dict['verbose']=bool(args[11]);
    dict['max_iter']=int(args[12]);
    dict['decision_function_shape']=args[13];
    dict['break_ties']=bool(args[14]);
    if args[15]=='None':
       args[15]=None
    dict['random_state']=args[15];
    dict['in_file_path']=args[16];
    dict['out_file_path']=args[17];
    return dict

def createDataSet(dict):
    dataset = pd.read_csv(dict['in_file_path'])#usecols = [3,4]
    header=dataset.columns.values;
    print(header)
    length=len(header)
    X = dataset.iloc[:,[length-3,length-2]].values
    #vec = DictVectorizer(sparse=False)
    #print(X0.to_dict(orient='record'))
    #X = vec.fit_transform(X0.to_dict(orient='record'))
    #X=str_column_to_int(X,[0])
    #fs_tsne=TSNE(n_components=2)
    #X = fs_tsne.fit_transform(X)


    Y0 = dataset.iloc[:,[length-1]].values
    Y0=str_column_to_int(Y0,[0])
    Y=[]
    for i in range (0,len(Y0)):
        Y.append(Y0[i][0])

    clf = svm.SVC(gamma='auto')
    clf.fit(X, Y)
    predict=clf.predict(X)
    print(predict)

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 500),
                     np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 500))

    # plot the decision function for each datapoint on the grid
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.show()



# 定义一个函数将数据集转化为可处理的数值
def str_column_to_int(dataset, columns):
    """
    将类别转化为int型
    @dataset: 数据
    @column: 需要转化的列
    """
    for column in columns:
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
    return dataset

if __name__ == "__main__":
    main()
