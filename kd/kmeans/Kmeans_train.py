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
from sklearn.preprocessing import LabelEncoder

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
    dict['n_clusters']=int(args[1]);
    dict['init']=args[2];
    dict['n_init']=int(args[3]);
    dict['max_iter']=int(args[4]);
    dict['tol']=float(args[5])
    dict['precompute_distances']=args[6];
    dict['verbose']=args[7];
    if args[8]=='None':
       args[8]=None
    dict['random_state']=args[8];
    dict['copy_x']=False;
    if args[9]=='True':
       dict['copy_x']=True
    if args[10]=='None':
       args[10]=None
    dict['n_jobs']=args[10];
    dict['algorithm']=args[11];
    dict['dimension']=int(args[12]);
    dict['in_file_path']=args[13];
    dict['out_file_path']=args[14];
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

    #print(X)

    #pca_sk = PCA(n_components=2)
    #数组降维
    if(dimension>=3):
        fs_tsne=TSNE(n_components=2)
        X = fs_tsne.fit_transform(X)

    clf=KMeans(
             n_clusters=dict['n_clusters'],
             init=dict['init'],
             n_init=dict['n_init'],
             max_iter=dict['max_iter'],
             tol=dict['tol'],
             precompute_distances=dict['precompute_distances'],
             verbose=dict['verbose'],
             random_state=dict['random_state'],
             copy_x=dict['copy_x'],
             n_jobs=dict['n_jobs'],
             algorithm=dict['algorithm']
    )
    #scaler=StandardScaler()#标准化
    #pipeline=make_pipeline(scaler,kmeans)

    predict=clf.fit_predict(X)
    label_pred = clf.labels_
    print(predict)
    centers = clf.cluster_centers_
    print(centers)

    model_path=dict['out_file_path']
    model_parent_path=os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    joblib.dump(clf, model_path)

    png_path=os.path.splitext(model_path)[0]+".png"
    csv_path=os.path.splitext(model_path)[0]+".csv"

    dataset['type']=predict
    dataset.to_csv(csv_path)

    plt.scatter(np.array(X)[:, 0], np.array(X)[:, 1], c=predict)
    plt.scatter(centers[:, 0], centers[:, 1], c="r")
    plt.savefig(png_path)

if __name__ == "__main__":
    main()
