print(__doc__)


import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.externals import joblib
import os

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
    dict['dimension']=int(args[1]);
    dict['model_file_path']=args[2];
    dict['in_file_path']=args[3];
    dict['out_file_path']=args[4];
    return dict

def createDataSet(dict):
    in_file_path=dict['in_file_path']
    dimension=dict['dimension']
    dataset = pd.read_csv(in_file_path)#usecols = [3,4]
    header=dataset.columns.values;
    length=len(header)
    #X = dataset.values
    header=dataset.columns.values;
    print(header)
    list =[]
    for i in range(length-dimension, length):
          list.append(i)
    X = dataset.iloc[:,list].values
    scla=StandardScaler()
    X=scla.fit_transform(X)
    #pca_sk = PCA(n_components=2)
    #数组降维
    if(dimension>=3):
        fs_tsne=TSNE(n_components=2)
        X = fs_tsne.fit_transform(X)

    clf = joblib.load(dict['model_file_path'])
    predict=clf.fit_predict(X)
    label_pred = clf.labels_
    print(predict)
    centers = clf.cluster_centers_
    print(centers)

    out_path=dict['out_file_path']
    out_parent_path=os.path.split(out_path)[0]
    if not os.path.exists(out_parent_path):
        os.makedirs(out_parent_path)

    dataset['type']=predict
    dataset.to_csv(out_path)

    png_path=os.path.splitext(out_path)[0]+".png"

    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=predict)
    plt.scatter(centers[:, 0], centers[:, 1], c="r")
    plt.savefig(png_path)

def modiData(data):
    x1 = []
    x2=[]
    for i in range(0,len(data+1)):
        x1.append(data[i][0])
        x2.append(data[i][1])
    x1=np.array(x1)
    x2=np.array(x2)
    #重塑数据
    X=np.array(list(zip(x1,x2))).reshape(len(x1),2)
    return X
if __name__ == "__main__":
    main()
