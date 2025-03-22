import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # 3x3 Convolutional Branch
    conv_3x3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pool_3x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_3x3)

    # 1x1 Convolutional Branch
    conv_1x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_3x3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    max_pool_1x1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_3x3)

    # Max Pooling Branch
    max_pool_3x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)

    # Concatenate outputs from different branches
    branch_outputs = Concatenate()([conv_3x3, conv_1x1, max_pool_1x1, max_pool_3x3])

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(branch_outputs)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model