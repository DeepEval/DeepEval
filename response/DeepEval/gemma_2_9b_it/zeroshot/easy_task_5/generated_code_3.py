import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

def dl_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=10, activation='softmax'))
    
    return model