import numpy as np
from assignment1 import *
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# x_min, x_max = -0.5, 1.5
# y_min, y_max = -0.5, 1.5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                      np.arange(y_min, y_max, 0.02))



X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])
y = y.reshape(-1, 1)
# print(np.c_[xx.ravel(), yy.ravel()])



l1 = Layer(2, 8, activation=relu)
l2 = Layer(8, 1, activation=sigmoid)
print(X.shape)
print(y.shape)
m = Model([l1, l2])
for i in range(10000):
		print(i)
		m.backprop_bin_cross_entropy(X, y, alpha= 0.03)

out1 = m.forward_pass(X)
print(out1.shape)
l = m.mean_squared_loss(out1, y)
print(l)

fig, ax = plt.subplots(5, 5, figsize=(12, 12), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

for i, axi in enumerate(ax.flat):


    # model.fit(X, y, nb_epoch=250)
    temp = np.c_[xx.ravel(), yy.ravel()]
    # print(temp.shape)
    Z = m.forward_pass(temp)
    # print(xx.shape)
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    axi.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axi.scatter(x=[0,1],y=[0,1], cmap=plt.cm.coolwarm, s=20)
    axi.scatter(x=[0,1],y=[1,0], cmap=plt.cm.coolwarm, s=20)

plt.show()



