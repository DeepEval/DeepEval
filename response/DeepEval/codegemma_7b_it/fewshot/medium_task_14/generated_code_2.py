import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate, Flatten, Dense, add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_act = Activation('relu')(block1_bn)
    block1_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block1_act)

    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(block1_pool)
    block2_bn = BatchNormalization()(block2_conv)
    block2_act = Activation('relu')(block2_bn)
    block2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(block2_act)

    # Block 3
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(block2_pool)
    block3_bn = BatchNormalization()(block3_conv)
    block3_act = Activation('relu')(block3_bn)

    # Parallel Branch
    parallel_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    parallel_conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    parallel_conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same')(input_layer)
    parallel_conv4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Output Paths
    path1 = add([block3_act, parallel_conv1])
    path2 = add([block3_act, parallel_conv2])
    path3 = add([block3_act, parallel_conv3])
    path4 = add([block3_act, parallel_conv4])

    # Classification
    flatten = Flatten()(path1)
    dense1 = Dense(units=64, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model