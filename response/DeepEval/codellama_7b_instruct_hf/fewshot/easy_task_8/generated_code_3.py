import keras
from keras.models import Sequential
from keras.layers import Input, DepthwiseConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    model = Sequential()
    model.add(DepthwiseConv2D(kernel_size=3, strides=1, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(kernel_size=1, strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(rate=0.2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=10, activation='softmax'))
    return model