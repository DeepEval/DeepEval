import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolutions followed by average pooling
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)
    block1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1)
    block1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1)

    # Path 2: Single convolutional layer
    block2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Flatten the outputs from both pathways
    flattened = Concatenate()([block1, block2])

    # Batch normalization and fully connected layers
    flattened = BatchNormalization()(flattened)
    flattened = Flatten()(flattened)
    flattened = Dense(units=128, activation='relu')(flattened)
    flattened = Dense(units=10, activation='softmax')(flattened)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=flattened)
    return model