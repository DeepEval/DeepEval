import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = ReLU()(block1_bn)

    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1_relu)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = ReLU()(block2_bn)

    # Block 3
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2_relu)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = ReLU()(block3_bn)

    # Parallel Branch
    parallel_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Output Paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(block3_relu)
    path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(block3_relu)
    path3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same')(block3_relu)

    # Add Paths
    concat = Add()([path1, path2, path3, parallel_conv])

    # Fully Connected Layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model