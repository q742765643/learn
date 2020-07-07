print(__doc__)

import sys
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.externals import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn import preprocessing

import os
#main 方法 传入参数
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
    dict['model_file_path']=args[1];
    dict['in_file_path']=args[2];
    dict['out_file_path']=args[3];
    return dict

def createDataSet(dict):

    #X_pred = pd.read_csv(dict['in_file_path'])

    #headers = X_pred.values.tolist()

    #print("headers\n",headers)
    #feature = X_pred[headers[:len(headers)]]
    '''
    vec = DictVectorizer()
    dummyX = vec.fit_transform(feature.to_dict(orient='record'))
    print('show vector name\n',vec.get_feature_names())
    clf = joblib.load(dict['model_file_path'])
    '''
    allElectronicsData = open(dict['in_file_path'])
    reader = csv.reader(allElectronicsData)
    headers = next(reader)

    print("headers\n",headers)
    featureList = []
    for row in [rows for rows in reader]:
            rowDict = {}
            for i in range(0, len(row)):
                rowDict[headers[i]] = row[i]
            featureList.append(rowDict)

    clf = joblib.load(dict['model_file_path'])

    vec = DictVectorizer()
    dummyX = vec.fit_transform(featureList).toarray()


    y_predict = clf.predict(dummyX)
    y_predict_proba = clf.predict_proba(dummyX)

    print(y_predict)
    print(y_predict_proba)
    y_predict_proba_List = []
    for i in range(0, len(y_predict_proba) ):
         y_predict_proba_List.append(str(clf.classes_)+str(y_predict_proba[i]))
    X_pred = pd.read_csv(dict['in_file_path'])
    X_pred['y_predict']=y_predict
    X_pred['y_predict_proba']=y_predict_proba_List

    out_path=dict['out_file_path']
    out_parent_path=os.path.split(out_path)[0]

    if not os.path.exists(out_parent_path):
        os.makedirs(out_parent_path)
    X_pred.to_csv(out_path,index=0)


if __name__ == "__main__":
    main()
