import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    dropout1 = Dropout(rate=0.25)(maxpool1)

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(dropout1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    dropout2 = Dropout(rate=0.25)(maxpool2)

    # Third convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    maxpool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    dropout3 = Dropout(rate=0.25)(maxpool3)

    # Restore number of channels
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(dropout3)
    flatten = Flatten()(conv4)
    dense = Dense(units=128, activation='relu')(flatten)
    dropout4 = Dropout(rate=0.5)(dense)
    output = Dense(units=10, activation='softmax')(dropout4)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model