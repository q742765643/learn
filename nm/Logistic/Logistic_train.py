print(__doc__)

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def main():
    print('参数个数为:', len(sys.argv), '个参数。')
    print('参数列表:', str(sys.argv))
    print('脚本名为：', sys.argv[0])
    for i in range(1, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))
    dict = createDict(sys.argv)
    createDataSet(dict)


def createDict(args):
    dict = {};
    dict['penalty'] = args[1];
    dict['dual'] = False;
    if args[2] == 'True':
        dict['dual'] = True
    print(dict)
    dict['tol'] = float(args[3]);
    dict['C'] = float(args[4]);
    dict['fit_intercept'] = False
    if args[5] == 'True':
        dict['fit_intercept'] = True
    dict['intercept_scaling'] = int(args[6]);
    if args[7] == 'None':
        args[7] = None
    dict['class_weight'] = args[7];
    if args[8] == 'None':
        args[8] = None
    dict['random_state'] = args[8];
    dict['solver'] = args[9];
    dict['max_iter'] = int(args[10]);
    dict['multi_class'] = args[11];
    dict['verbose'] = int(args[12]);
    dict['warm_start'] = False;
    if args[13] == 'True':
        dict['warm_start'] = True
    if args[14] == 'None':
        args[14] = None
    dict['n_jobs'] = args[14];
    if args[15] == 'None':
        args[15] = None
    dict['l1_ratio'] = args[15];
    dict['dimension'] = int(args[16]);
    dict['in_file_path'] = args[17];
    dict['out_file_path'] = args[18];
    return dict


def createDataSet(dict):
    dataset = pd.read_csv(dict['in_file_path'])  # usecols = [3,4]
    header = dataset.columns.values;
    length = len(header)
    dimension = dict['dimension']

    list = []
    for i in range(length - dimension - 1, length - 1):
        list.append(i)
    X = dataset.iloc[:, list].values
    scla = StandardScaler()
    X = scla.fit_transform(X)
    if (dimension > 2):
        X = PCA(n_components=2).fit_transform(X)
    Y0 = dataset.iloc[:, [length - 1]].values
    Y = []
    for i in range(0, len(Y0)):
        Y.append(Y0[i][0])
    Y = np.array(Y)
    print(Y)

    clf = LogisticRegression(penalty=dict['penalty'],
                             dual=dict['dual'],
                             tol=dict['tol'],
                             C=dict['C'],
                             fit_intercept=dict['fit_intercept'],
                             intercept_scaling=dict['intercept_scaling'],
                             class_weight=dict['class_weight'],
                             random_state=dict['random_state'],
                             solver=dict['solver'],
                             max_iter=dict['max_iter'],
                             multi_class=dict['multi_class'],
                             verbose=dict['verbose'],
                             warm_start=dict['warm_start'],
                             n_jobs=dict['n_jobs'],
                             l1_ratio=dict['l1_ratio']
                             )
    clf.fit(X, Y)
    # print the training scores
    print("training score : %.3f " % (clf.score(X, Y)))

    model_path = dict['out_file_path']
    model_parent_path = os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    joblib.dump(clf, model_path)

    # create a mesh to plot in
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    plt.title("Decision surface of LogisticRegression (%s)")
    plt.axis('tight')

    # Plot also the training points
    for i in zip(clf.classes_):
        idx = np.where(Y == i)
        plt.scatter(X[idx, 0], X[idx, 1], cmap=plt.cm.Paired,
                    edgecolor='black', s=20)

    png_path = os.path.splitext(model_path)[0] + ".png"
    plt.savefig(png_path)


if __name__ == "__main__":
    main()
