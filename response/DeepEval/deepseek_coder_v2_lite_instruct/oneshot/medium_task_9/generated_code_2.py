import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, AveragePooling2D
from keras.models import Model

def basic_block(input_tensor):
    # Main path
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Branch path
    branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
    branch = BatchNormalization()(branch)
    branch = ReLU()(branch)

    # Feature fusion
    output_tensor = Add()([x, branch])
    output_tensor = ReLU()(output_tensor)

    return output_tensor

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer to reduce dimensionality
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # First block
    block1 = basic_block(x)
    block1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(block1)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    # Second block
    block2 = basic_block(block1)
    block2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(block2)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    # Average pooling layer
    avg_pool = AveragePooling2D(pool_size=(8, 8), strides=1)(block2)
    flatten_layer = Flatten()(avg_pool)

    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model