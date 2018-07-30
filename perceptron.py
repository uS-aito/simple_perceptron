import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(input_dim=2, units=1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

X = np.array( [[0,0],
              [0,1],
              [1,0],
              [1,1]] )
T = np.array( [[0],
               [1],
               [1],
               [1]] )

model.fit(X, T, epochs=30, batch_size=1)
Y = model.predict_classes(X, batch_size=1)

print(Y == T)
