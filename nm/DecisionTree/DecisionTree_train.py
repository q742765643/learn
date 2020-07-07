print(__doc__)

import csv
import os
import sys

import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import plot_tree


# main 方法 传入参数
def main():
    print('参数个数为:', len(sys.argv), '个参数。')
    print('参数列表:', str(sys.argv))
    print('脚本名为：', sys.argv[0])
    for i in range(1, len(sys.argv)):
        print('参数 %s 为：%s' % (i, sys.argv[i]))
    dict = createDict(sys.argv)
    createDataSet(dict)


# 创建参数生成字典
def createDict(args):
    dict = {};
    dict['criterion'] = args[1];
    dict['splitter'] = args[2];
    if args[3] == 'None':
        args[3] = None
    dict['max_depth'] = args[3];
    dict['min_samples_split'] = int(args[4])
    dict['min_samples_leaf'] = int(args[5]);
    dict['min_weight_fraction_leaf'] = float(args[6]);
    if args[7] == 'None':
        args[7] = None
    dict['max_features'] = args[7];
    if args[8] == 'None':
        args[8] = None
    dict['random_state'] = args[8];
    if args[9] == 'None':
        args[9] = None
    dict['max_leaf_nodes'] = args[9];
    dict['min_impurity_decrease'] = float(args[10]);
    if args[11] == 'None':
        args[11] = None
    dict['min_impurity_split'] = args[11];
    if args[12] == 'None':
        args[12] = None
    dict['class_weight'] = args[12];
    dict['presort'] = args[13];
    dict['ccp_alpha'] = float(args[14]);
    dict['in_file_path'] = args[15];
    dict['out_file_path'] = args[16];
    return dict


def createDataSet(dict):
    allElectronicsData = open(dict['in_file_path'])
    reader = csv.reader(allElectronicsData)
    headers = next(reader)

    print("headers\n", headers)
    featureList = []

    labelList = []

    for row in [rows for rows in reader]:
        labelList.append(row[len(row) - 1])
        rowDict = {}
        for i in range(0, len(row) - 1):
            rowDict[headers[i]] = row[i]
        featureList.append(rowDict)

    print("labelList\n", labelList)
    vec = DictVectorizer()
    dummyX = vec.fit_transform(featureList).toarray()
    print('show vector name\n', vec.get_feature_names())

    print("dummyX\n", dummyX)
    # 把标签转化为0-1形式
    # lb = preprocessing.LabelBinarizer()
    # dummyY = lb.fit_transform(labelList)
    dummyY = labelList
    print("dummyY\n", dummyY)
    print(dict)
    clf = tree.DecisionTreeClassifier(
        criterion=dict['criterion'],
        splitter=dict['splitter'],
        max_depth=dict['max_depth'],
        min_samples_split=dict['min_samples_split'],
        min_samples_leaf=dict['min_samples_leaf'],
        min_weight_fraction_leaf=dict['min_weight_fraction_leaf'],
        max_features=dict['max_features'],
        random_state=dict['random_state'],
        max_leaf_nodes=dict['max_leaf_nodes'],
        min_impurity_decrease=dict['min_impurity_decrease'],
        min_impurity_split=dict['min_impurity_split'],
        class_weight=dict['class_weight'],
        presort=dict['presort'],
        ccp_alpha=dict['ccp_alpha'])
    print(clf)
    clf.fit(dummyX, dummyY)
    print("training score : %.3f " % (clf.score(dummyX, dummyY)))
    import pydotplus
    from six import StringIO
    dot_data = StringIO()
    model_path = dict['out_file_path']
    model_parent_path = os.path.split(model_path)[0]
    if not os.path.exists(model_parent_path):
        os.makedirs(model_parent_path)
    pdf_path = os.path.splitext(model_path)[0] + ".pdf"
    dot_path = os.path.splitext(model_path)[0] + ".dot"
    with open(dot_path, 'w') as f:
        f = tree.export_graphviz(clf, out_file=f, class_names=clf.classes_,
                                 feature_names=vec.get_feature_names())
    tree.export_graphviz(clf, out_file=dot_data, feature_names=vec.get_feature_names(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf(pdf_path)

    joblib.dump(clf, model_path)
    plt.figure()
    plot_tree(clf, filled=True, feature_names=vec.get_feature_names(), class_names=clf.classes_)
    png_path = os.path.splitext(model_path)[0] + ".png"
    plt.savefig(png_path)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())


if __name__ == "__main__":
    main()
