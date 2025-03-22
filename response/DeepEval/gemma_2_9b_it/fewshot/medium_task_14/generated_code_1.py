import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1_1 = BatchNormalization()(conv1_1)
    relu1_1 = ReLU()(bn1_1)

    # Block 2
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1_1)
    bn2_1 = BatchNormalization()(conv2_1)
    relu2_1 = ReLU()(bn2_1)

    # Block 3
    conv3_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2_1)
    bn3_1 = BatchNormalization()(conv3_1)
    relu3_1 = ReLU()(bn3_1)

    # Parallel branch
    conv_parallel = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn_parallel = BatchNormalization()(conv_parallel)
    relu_parallel = ReLU()(bn_parallel)

    # Concatenate all outputs
    merged_features = Add()([relu3_1, relu_parallel])

    flatten = Flatten()(merged_features)

    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model