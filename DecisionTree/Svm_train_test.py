from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np

#生成数据
x0 = np.random.randint(0, 10, [50, 2])  #第一类点：在[0,0]到[45,45]之间的正方形内
x1 = np.random.randint(-10, 0, [50, 2])#第二类点：在[55,55]到[100,100]之间的正方形内
X = np.concatenate([x0, x1])            #合并矩阵

y0 = np.zeros(50)    #第一类标签，以0表示
y1 = np.ones(50)     #第二类标签，以1表示
Y = np.concatenate([y0, y1]) #合并
Y = np.logical_xor(X[:, 0] > 30, X[:, 1] > 0)

#SVM
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)



#获取超平面y=kx+b的k和b
weight = clf.coef_[0] #取出权重矩阵
bias = clf.intercept_[0]
# w0 * x + w1 * y + bias = 0
# y = - w0/w1 * x - bias / w1
k = -weight[0] / weight[1]
b = -bias / weight[1]

#画图
support_vector = clf.support_vectors_
#画出散点图
plt.scatter(x0.T[0], x0.T[1], c='b')
plt.scatter(x1.T[0], x1.T[1], c='g')
#画出支持向量
plt.scatter(support_vector.T[0][0], support_vector.T[1][0], marker=',', c='r')
plt.scatter(support_vector.T[0][1], support_vector.T[1][1], marker=',', c='r')
#画出超平面
x = np.linspace(0, 100)
y = k * x + b
plt.plot(x, y)
plt.show()

xx, yy = np.meshgrid(np.linspace(-45, 45, 500),
                     np.linspace(-45, 45, 500))

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
plt.axis([-45, 45, -45, 45])
plt.show()


