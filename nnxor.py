from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

def get_model():
    model = Sequential()
    model.add(Dense(8, activation='relu', input_dim=2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    return model


X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0,1,1,0])

model = get_model()



fig, ax = plt.subplots(5, 5, figsize=(12, 12), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

for i, axi in enumerate(ax.flat):


    model.fit(X, y, nb_epoch=250)

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    axi.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    axi.scatter(x=[0,1],y=[0,1], cmap=plt.cm.coolwarm, s=20)
    axi.scatter(x=[0,1],y=[1,0], cmap=plt.cm.coolwarm, s=20)

plt.show()
