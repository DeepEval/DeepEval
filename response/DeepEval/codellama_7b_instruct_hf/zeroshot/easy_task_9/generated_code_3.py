from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    model = Sequential()
    model.add(Conv2D(32, (1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model