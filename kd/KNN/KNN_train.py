print(__doc__)

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn import neighbors

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
    dict['n_neighbors']=int(args[1])
    dict['weights']=args[2]
    dict['algorithm']=args[3]
    dict['leaf_size']=int(args[4])
    dict['p']=int(args[5])
    dict['metric']=args[6]
    if args[7]=='None':
         args[7]=None
    dict['metric_params']=args[7]
    if args[8]=='None':
         args[8]=None
    dict['n_jobs']=args[8]
    dict['dimension']=int(args[9]);
    dict['in_file_path']=args[10];
    dict['out_file_path']=args[11];
    return dict

def createDataSet(dict):
    dataset = pd.read_csv(dict['in_file_path'])#usecols = [3,4]
    header=dataset.columns.values;
    length=len(header)
    dimension=dict['dimension']

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
    clf = neighbors.KNeighborsClassifier(n_neighbors=dict['n_neighbors'],
                 weights=dict['weights'], algorithm=dict['algorithm'],
                 leaf_size=dict['leaf_size'],p=dict['p'],
                 metric=dict['metric'], metric_params=dict['metric_params'],
                 n_jobs=dict['n_jobs'])
    clf.fit(X, Y)
    h = .02
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=Y,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (dict['n_neighbors'], dict['weights']))
    print("training score : %.3f " % (clf.score(X, Y)))

    model_path=dict['out_file_path']
    model_parent_path=os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    joblib.dump(clf, model_path)
    png_path=os.path.splitext(model_path)[0]+".png"
    plt.savefig(png_path)

if __name__ == "__main__":
    main()
