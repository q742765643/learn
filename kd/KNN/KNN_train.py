print(__doc__)

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
def main():

    print('参数个数为:', len(sys.argv), '个参数。')
    print('参数列表:', str(sys.argv))
    print('脚本名为：', sys.argv[0])
    for i in range(1, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))

    dict=createDict(sys.argv)

    createDataSet(dict)



# 创建参数生成字典
def createDict(args):
    dict={};
    return dict

def createDataSet(dict):
    print(dict)
    dataset = pd.read_csv('F:/机器学习/knn/train.csv')#usecols = [3,4]
    header=dataset.columns.values;
    length=len(header)
    dimension=2

    list =[]
    for i in range(length-dimension-1, length-1):
          list.append(i)

    X = dataset.iloc[:,list].values
    scla=StandardScaler()
    X=scla.fit_transform(X)
    if(dimension>2):
      X = PCA(n_components=2).fit_transform(X)

    Y0 = dataset.iloc[:,[length-1]].values
    Y=[]
    for i in range (0,len(Y0)):
        Y.append(Y0[i][0])
    Y=np.array(Y)


