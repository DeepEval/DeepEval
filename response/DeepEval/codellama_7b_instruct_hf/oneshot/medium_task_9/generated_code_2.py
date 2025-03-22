import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)

    # Feature fusion
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = ReLU()(conv2)
    output = Add()([conv1, conv2])

    # Main structure
    block1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(output)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1)

    block2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block2)

    # Output
    output = Flatten()(block2)
    output = Dense(units=10, activation='softmax')(output)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model