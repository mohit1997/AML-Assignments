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
y = np.array([0,1,1,0])

model = get_model()



fig, ax = plt.subplots(5, 5, figsize=(12, 12), subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))


for i, axi in enumerate(ax.flat):
    model.fit(X, y, nb_epoch=300, verbose=1)

    axi.scatter(x=[0,1],y=[0,1])
    axi.scatter(x=[0,1],y=[1,0])


    w1 = model.layers[0].get_weights()[0][0]
    w2 = model.layers[0].get_weights()[0][1]
    b = model.layers[0].get_weights()[1]

    a = -w1 / w2
    xx = np.linspace(-.5, 1.5)
    yy = a * xx - b / w2

    axi.plot(xx, yy, 'k-')

    print w1, w2, b

    """
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    axi.pcolormesh(xx, yy, Z, cmap=cmap_light)
    """

plt.show()
