import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path="ex2data1.txt"
data=pd.read_csv(path,header=None,names=["exam1","exam2","admission"])
# print(data.head())

positive=data[data["admission"].isin([1])]
negative=data[data["admission"].isin([0])]

fig,ax=plt.subplots(figsize=(10,8))#ax为主要做图区，做图命令都用ax
ax.scatter(positive["exam1"],positive["exam2"],marker="o",color="red",label="positive")
ax.scatter(negative["exam1"],negative["exam2"],marker="o",color="blue",label="negative")

ax.legend()#图例
plt.xlabel("exam1")
plt.ylabel("exam2")
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

# x=np.arange(-10,10,0.1)
# y=sigmoid(x)
# plt.plot(x,y)
# plt.show()
