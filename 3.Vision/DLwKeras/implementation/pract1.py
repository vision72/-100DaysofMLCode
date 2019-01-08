from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
np.random.seed(1671)


epochs = 10
batch = 128
verbose = 1
classes = 10
optimizer = SGD()
hid_lay = 128
valid_portion = 0.2


(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 28 x 28
reshaped = 784
X_train = X_train.reshape(60000, reshaped)
X_test = X_test.reshape(10000, reshaped)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing
X_train /= 255
X_test	/= 255

print('Train Samples : {}'.format(X_train.shape[0]))
print('Test Samples  : {}'.format(X_test.shape[0]))

y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

model = Sequential()
model.add(Dense(hid_lay, input_shape=(reshaped,)))
model.add(Activation('relu'))
model.add(Dense(hid_lay))
model.add(Activation('relu'))
model.add(Dense(classes))
model.add(Activation('softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
	metrics=['accuracy'])
history = model.fit(X_train, y_train,
	batch_size=batch, epochs=epochs,
	verbose=verbose, validation_split=valid_portion)
score = model.evaluate(X_test, y_test, verbose=verbose)
print('Test Score:    {}'.format(score[0]))
print('Test Accuracy: {}'.format(score[1]))
