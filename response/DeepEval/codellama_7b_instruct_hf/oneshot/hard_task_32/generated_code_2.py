import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1: Depthwise Separable Convolutional Layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)

    # Branch 2: 1x1 Convolutional Layer
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)
    branch2 = Dropout(rate=0.2)(branch2)

    # Branch 3: 5x5 Convolutional Layer
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch3)
    branch3 = Dropout(rate=0.2)(branch3)

    # Concatenate the outputs from the three branches
    x = Concatenate()([branch1, branch2, branch3])

    # Batch Normalization
    x = BatchNormalization()(x)

    # Flatten the output
    x = Flatten()(x)

    # Fully Connected Layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=x)
    return model