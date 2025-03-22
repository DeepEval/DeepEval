import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_output = block1_bn

    # Block 2
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block2_bn = BatchNormalization()(block2_conv)
    block2_output = block2_bn

    # Block 3
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block3_bn = BatchNormalization()(block3_conv)
    block3_output = block3_bn

    # Parallel branch
    parallel_conv1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    parallel_conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(parallel_conv1)
    parallel_conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(parallel_conv2)
    parallel_output = Concatenate()([parallel_conv1, parallel_conv2, parallel_conv3])

    # Add all paths
    output_tensor = Add()([block3_output, parallel_output])

    # Flatten and classification
    flatten_layer = Flatten()(output_tensor)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model