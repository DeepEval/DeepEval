import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense, Concatenate

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1_1 = BatchNormalization()(conv1_1)
    relu1_1 = ReLU()(batch_norm1_1)

    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1_1)
    batch_norm1_2 = BatchNormalization()(conv1_2)
    relu1_2 = ReLU()(batch_norm1_2)

    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1_2)
    batch_norm1_3 = BatchNormalization()(conv1_3)
    relu1_3 = ReLU()(batch_norm1_3)

    # Block 2
    conv2_1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu1_3)
    batch_norm2_1 = BatchNormalization()(conv2_1)
    relu2_1 = ReLU()(batch_norm2_1)

    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2_1)
    batch_norm2_2 = BatchNormalization()(conv2_2)
    relu2_2 = ReLU()(batch_norm2_2)

    conv2_3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2_2)
    batch_norm2_3 = BatchNormalization()(conv2_3)
    relu2_3 = ReLU()(batch_norm2_3)

    # Block 3
    conv3_1 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu2_3)
    batch_norm3_1 = BatchNormalization()(conv3_1)
    relu3_1 = ReLU()(batch_norm3_1)

    conv3_2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu3_1)
    batch_norm3_2 = BatchNormalization()(conv3_2)
    relu3_2 = ReLU()(batch_norm3_2)

    conv3_3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(relu3_2)
    batch_norm3_3 = BatchNormalization()(conv3_3)
    relu3_3 = ReLU()(batch_norm3_3)

    # Parallel branch
    conv_parallel = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_parallel = BatchNormalization()(conv_parallel)
    relu_parallel = ReLU()(batch_norm_parallel)

    # Concatenate outputs
    concat = Concatenate()([relu1_3, relu2_3, relu3_3, relu_parallel])

    # Fully connected layers
    flatten_layer = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model