print(__doc__)

import sys
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm,datasets
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
    dict['dimension']=int(args[16]);
    dict['in_file_path']=args[17];
    dict['out_file_path']=args[18];
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
    if(dimension>2):
      X = PCA(n_components=2).fit_transform(X)

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
    print(Y)
    clf = svm.SVC(
                  C=dict['C'],
                  kernel=dict['kernel'],
                  degree=dict['degree'],
                  gamma=dict['gamma'],
                  coef0=dict['coef0'],
                  shrinking=dict['shrinking'],
                  probability=dict['probability'],
                  tol=dict['tol'],
                  cache_size=dict['cache_size'],
                  class_weight=dict['class_weight'],
                  verbose=dict['verbose'],
                  max_iter=dict['max_iter'],
                  decision_function_shape=dict['decision_function_shape'],
                  break_ties=dict['break_ties'],
                  random_state=dict['random_state']
                  )
    print(clf)
    clf.fit(X, Y)
    model_path=dict['out_file_path']
    model_parent_path=os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    joblib.dump(clf, model_path)

    fig,ax = plt.subplots()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())

    png_path=os.path.splitext(model_path)[0]+".png"
    plt.savefig(png_path)


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out
    '''
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
    '''


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
