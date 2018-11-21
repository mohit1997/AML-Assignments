from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

def get_model():
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=2))
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([0,1,1,0])

model = get_model()



fig, ax = plt.subplots(5, 5, figsize=(12, 12), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


for i, axi in enumerate(ax.flat):
    model.fit(X, Y, nb_epoch=10, verbose=1)
    
    w1 = model.layers[0].get_weights()[0][0]
    w2 = model.layers[0].get_weights()[0][1]
    b = model.layers[0].get_weights()[1]

    a = -w1 / w2
    x = np.linspace(-.5, 1.5)
    y = a * x - b / w2

    axi.plot(x, y, 'k-')

    print w1, w2, b

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape, xx.shape, yy.shape)
    Z = Z.reshape(xx.shape)
    axi.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axi.scatter(x=[0,1],y=[0,1], cmap=plt.cm.coolwarm, s=20)
    axi.scatter(x=[0,1],y=[1,0], cmap=plt.cm.coolwarm, s=20)
# axi.pcolormesh(xx, yy, Z, cmap=cmap_light)

plt.show()
