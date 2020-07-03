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
    dataset = pd.read_csv(dict['in_file_path'])#usecols = [3,4]
    header=dataset.columns.values;
    length=len(header)
    dimension=dict['dimension']

    list =[]
    for i in range(length-dimension, length):
          list.append(i)

    X = dataset.iloc[:,list].values
    scla=StandardScaler()
    X=scla.fit_transform(X)
    if(dimension>2):
      X = PCA(n_components=2).fit_transform(X)

    clf = joblib.load(dict['model_file_path'])
    predict=clf.predict(X)
    predict_proba = clf.predict_proba(X)
    out_path=dict['out_file_path']
    out_parent_path=os.path.split(out_path)[0]
    if not os.path.exists(out_parent_path):
        os.makedirs(out_parent_path)
    predict_proba_List = []
    for i in range(0, len(predict_proba) ):
         predict_proba_List.append(str(clf.classes_)+str(predict_proba[i]))
    print(predict_proba)
    dataset['predict']=predict
    dataset['predict_proba']=predict_proba_List
    dataset.to_csv(out_path)

if __name__ == "__main__":
    main()
