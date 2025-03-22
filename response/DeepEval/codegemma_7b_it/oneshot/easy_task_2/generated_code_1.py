import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Dropout

def dl_model():
    model = Sequential()

    # First Feature Extraction Layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(224, 224, 3)))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Second Feature Extraction Layer
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Additional Convolutional Layers and Average Pooling
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid'))

    # Flatten Feature Maps
    model.add(Flatten())

    # Fully Connected Layers with Dropout
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units=1024, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer for Classification Probabilities
    model.add(Dense(units=1000, activation='softmax'))

    return model