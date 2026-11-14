# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# path="ex2data1.txt"
# data=pd.read_csv(path,header=None,names=["exam1","exam2","admission"])
# # print(data.head())
#
# # positive=data[data["admission"].isin([1])]
# # negative=data[data["admission"].isin([0])]
# #
# # fig,ax=plt.subplots(figsize=(10,8))#ax为主要做图区，做图命令都用ax
# # ax.scatter(positive["exam1"],positive["exam2"],marker="o",color="red",label="positive")
# # ax.scatter(negative["exam1"],negative["exam2"],marker="o",color="blue",label="negative")
# #
# # ax.legend()#图例
# # plt.xlabel("exam1")
# # plt.ylabel("exam2")
# # plt.show()
#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# # x=np.arange(-10,10,0.1)
# # y=sigmoid(x)
# # plt.plot(x,y)
# # plt.show()
#
# def cost_func(theta, X, y):
#     theta = theta.reshape(-1, 1)
#     loss = np.sum(np.log(sigmoid(X@theta))*y*(-1)-(1-y)*np.log(1-sigmoid(X@theta)))/y.shape[0]
#     print(sigmoid(X@theta).shape)
#     return loss
#
# data.insert(0,"ones",value=1)
# print(data.head())
# # print(data.shape)
# cols=data.shape[1]
# X=data.iloc[:,0:cols-1]
# y=data.iloc[:,cols-1:cols]
# X=np.array(X.values)
# y=np.array(y.values)
# # #
# theta=np.zeros((3,1))
# #
# # print(cost_func(theta,X,y))
#
# def gradient(theta,X,y):
#     theta = theta.reshape(-1, 1)
#     grad=np.zeros(3)
#     for i in range(theta.reshape(-1,1).shape[0]):
#         grad[i]=np.sum((sigmoid(X @ theta)-y)*X[:,i].reshape(-1,1))/X.shape[0]
#     return grad
#
# grad=gradient(theta,X,y)
# # print(grad)
# theta=np.zeros(3)
# import scipy.optimize as opt
# result=opt.fmin_tnc(func=cost_func,x0=theta,fprime=gradient,args=(X,y))
# print(result)
# print(cost_func(result[0],X,y))
#
# def predict(theta,X):
#     theta = theta.reshape(-1,1)
#     sig_value=sigmoid(X @ theta)
#     # compare_arr=0.5*np.ones(sig_value.shape)
#     return sig_value>=0.5
#
# pred=predict(result[0],X)
# # accuracy=np.sum((pred==y)/y.shape[0])
# accuracy=np.mean(pred==y)
# print(accuracy)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path="ex2data2.txt"
data=pd.read_csv(path,header=None,names=["test1","test2","accepted"])

positive=data[data["accepted"].isin([1])]
negative=data[data["accepted"].isin([0])]

fig,ax=plt.subplots(figsize=(10,5))
ax.scatter(positive["test1"],positive["test2"],label="positive")
ax.scatter(negative["test1"],negative["test2"],label="negative")

plt.legend()
plt.xlabel("test1")
plt.ylabel("test2")
plt.show()
