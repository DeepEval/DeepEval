import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1: Two blocks of convolution followed by average pooling
    def path1_block(input_tensor):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
        return x

    x1 = path1_block(input_layer)
    x1 = path1_block(x1)

    # Path 2: Single convolutional layer
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine outputs from both paths
    combined = Add()([x1, x2])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer to produce the final classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model