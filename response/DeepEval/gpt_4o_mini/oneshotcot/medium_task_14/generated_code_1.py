import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm1_1 = BatchNormalization()(conv1_1)
    relu1_1 = ReLU()(batch_norm1_1)

    # Block 2
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu1_1)
    batch_norm1_2 = BatchNormalization()(conv1_2)
    relu1_2 = ReLU()(batch_norm1_2)

    # Block 3
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(relu1_2)
    batch_norm1_3 = BatchNormalization()(conv1_3)
    relu1_3 = ReLU()(batch_norm1_3)

    # Parallel convolutional branch
    parallel_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    batch_norm_parallel = BatchNormalization()(parallel_conv)
    relu_parallel = ReLU()(batch_norm_parallel)

    # Adding outputs of blocks
    added_output = Add()([relu1_3, relu_parallel])

    # Flatten and fully connected layers
    flatten_layer = Flatten()(added_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model