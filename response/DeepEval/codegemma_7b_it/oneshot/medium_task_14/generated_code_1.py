import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, concatenate, Flatten, Dense, add

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1_block1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm1_block1 = BatchNormalization()(conv1_block1)
    relu1_block1 = ReLU()(batch_norm1_block1)
    max_pool1_block1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu1_block1)

    # Block 2
    conv1_block2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool1_block1)
    batch_norm1_block2 = BatchNormalization()(conv1_block2)
    relu1_block2 = ReLU()(batch_norm1_block2)
    max_pool1_block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu1_block2)

    # Block 3
    conv1_block3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same')(max_pool1_block2)
    batch_norm1_block3 = BatchNormalization()(conv1_block3)
    relu1_block3 = ReLU()(batch_norm1_block3)
    max_pool1_block3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(relu1_block3)

    # Parallel branch
    conv_parallel = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm_parallel = BatchNormalization()(conv_parallel)
    relu_parallel = ReLU()(batch_norm_parallel)

    # Output paths
    output_block1 = max_pool1_block1
    output_block2 = max_pool1_block2
    output_block3 = max_pool1_block3
    output_parallel = relu_parallel

    # Concatenate outputs and add parallel branch
    concat = concatenate([output_block1, output_block2, output_block3, output_parallel])
    add_layer = add([concat, concat])

    # Fully connected layers
    flatten_layer = Flatten()(add_layer)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model