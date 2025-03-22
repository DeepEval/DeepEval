import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout

def dl_model():
    model = Sequential()

    # Extracting spatial features
    model.add(Conv2D(32, (7, 7), padding='same', use_bias=False, input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Channel-wise feature transformation
    model.add(Conv2D(32, (1, 1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Combine original input with processed features
    model.add(Add()([model.layers[0].output, model.layers[3].output]))

    # Classification
    model.add(Flatten())
    model.add(Dense(10))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model