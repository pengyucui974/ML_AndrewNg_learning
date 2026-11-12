# import numpy as np
# # ans=np.eye(5,5,dtype=int)
# # print(ans)
# import matplotlib.pyplot as plt
# data=np.loadtxt('ex1data1.txt',delimiter=',')
# x=data[:,0]
# y=data[:,1]
# # plt.scatter(x,y,marker='o')
# # plt.xlabel("population of city in 10,000s")
# # plt.ylabel("profit in 10,000s")
# # plt.show()
# m=len(x)
# # cost_fun=0
# theta=np.zeros((2))
# x_predict=theta[0]+theta[1]*x
# # for i in range(m):
# #     cost_fun+=(theta[0]+theta[1]*x[i]-y[i])**2
# # cost_fun=cost_fun/(2*m)
# cost_fun_list=[]
# learning_rate=0.01
# cost_fun=np.mean((x_predict-y)**2)/2
# for i in range(10):
#     derivative_func_theta_0 = 0
#     derivative_func_theta_1 = 0
#     for j in range(m):
#         derivative_func_theta_1 += (theta[0] + theta[1] * x[j] - y[j])*x[j]
#         derivative_func_theta_0 += theta[0] + theta[1] * x[j] - y[j]
#     theta[0]=theta[0]-learning_rate*(derivative_func_theta_0)/m
#     theta[1]=theta[1]-learning_rate*(derivative_func_theta_1)/m
#     x_predict = theta[0] + theta[1] * x
#     cost_fun = np.mean((x_predict - y) ** 2) / 2
#     cost_fun_list.append(cost_fun)
#
# print(cost_fun_list)


# import numpy as np
# import matplotlib.pyplot as plt
# data=np.loadtxt("ex1data1.txt",delimiter=",")
# x=data[:,0]
# y=data[:,1]
# learning_rate=0.01
# theta=np.zeros(2)
# theta_0=[0]
# theta_1=[1]
# cost_func_list=[]
# for i in range(100):
#     error=theta[0]+theta[1]*x-y
#     gradient_0=np.mean(error)
#     gradient_1=np.mean(error*x)
#
#     theta[0]=theta[0]-learning_rate*gradient_0
#     theta_0.append(theta[0])
#     theta[1]=theta[1]-learning_rate*gradient_1
#     theta_1.append(theta[1])
#
#     error=theta[0]+theta[1]*x-y
#     cost_func_list.append(np.mean(error**2)/2)
# theta_0_values=np.linspace(-10,10,100)
# theta_1_values=np.linspace(-1,4,100)
# T0,T1=np.meshgrid(theta_0_values,theta_1_values)
# fig=plt.figure(size=(9,6))
# ax=fig.add_subplot(111,projection='3d')
# surface=ax.plot_surface(T0,T1,cost_func_list,cmap="viridis", alpha=0.8)
#
# import numpy as np
# import matplotlib.pyplot as plt
#
# data = np.loadtxt("ex1data1.txt", delimiter=",")
# x = data[:, 0]
# y = data[:, 1]
#
# learning_rate = 0.01
# theta = np.zeros(2)

# # 记录初始参数和初始损失，保证三个列表一一对应
# theta_0 = [theta[0]]
# theta_1 = [theta[1]]
# error = theta[0] + theta[1] * x - y
# cost_func_list = [np.mean(error ** 2) / 2]
#
# for i in range(15000):
#     error = theta[0] + theta[1] * x - y
#     gradient_0 = np.mean(error)
#     gradient_1 = np.mean(error * x)
#
#     theta[0] -= learning_rate * gradient_0
#     theta[1] -= learning_rate * gradient_1
#
#     theta_0.append(theta[0])
#     theta_1.append(theta[1])
#
#     error = theta[0] + theta[1] * x - y
#     cost_func_list.append(np.mean(error ** 2) / 2)
#
# # 生成参数网格
# theta_0_values = np.linspace(-10, 10, 100)
# theta_1_values = np.linspace(-1, 4, 100)
# T0, T1 = np.meshgrid(theta_0_values, theta_1_values)
#
# # 计算网格上每组参数的损失
# J = np.zeros_like(T0)
#
# for row in range(T0.shape[0]):
#     for col in range(T0.shape[1]):
#         error = T0[row, col] + T1[row, col] * x - y
#         J[row, col] = np.mean(error ** 2) / 2
#
# # 绘制损失曲面：参数名是 figsize
# fig = plt.figure(figsize=(9, 6))
# ax = fig.add_subplot(111, projection="3d")
#
# surface = ax.plot_surface(
#     T0, T1, J,
#     cmap="viridis",
#     alpha=0.6
# )
#
# # 绘制梯度下降轨迹
# ax.plot(
#     theta_0, theta_1, cost_func_list,
#     color="red",
#     linewidth=2,
#     label="Gradient descent"
# )
#
# # 标出起点和终点
# ax.scatter(
#     theta_0[0], theta_1[0], cost_func_list[0],
#     color="black", s=50, label="Start",
#     depthshade=False
# )
# ax.scatter(
#     theta_0[-1], theta_1[-1], cost_func_list[-1],
#     color="red", s=50, label="End",
#     depthshade=False
# )
#
# ax.set_xlabel(r"$\theta_0$")
# ax.set_ylabel(r"$\theta_1$")
# ax.set_zlabel("Cost")
# ax.legend()
#
# fig.colorbar(surface, ax=ax, shrink=0.6, label="Cost")
# plt.show()

import numpy as np
data=np.loadtxt("ex1data2.txt",delimiter=",")
x1=data[:,0]
x2=data[:,1]
y =data[:,2]
#feature scaling
x1_mean,x1_std=np.mean(x1),np.std(x1)
x2_mean,x2_std=np.mean(x2),np.std(x2)
x1=(x1-np.mean(x1))/np.std(x1)
x2=(x2-np.mean(x2))/np.std(x2)
theta=np.zeros(3)
gradient=np.zeros(3)
learning_rate=0.1
cost_func_list = []
for i in range(1000):
    error = theta[0] + theta[1] * x1 + theta[2] * x2 - y
    gradient[0] = np.mean(error)
    gradient[1] = np.mean(error*x1)
    gradient[2] = np.mean(error*x2)

    theta = theta - learning_rate * gradient
    error = theta[0] + theta[1] * x1 + theta[2] * x2 - y
    cost_func_list.append(np.mean(error**2)/2)
print(cost_func_list)
