import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Pathway 1: Convolutional blocks with average pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    avg_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Pathway 2: Single convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine pathways
    concatenate = Concatenate()([avg_pooling, conv3])

    # Flatten and map to probability distribution
    flatten = Flatten()(concatenate)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=dense)
    return model