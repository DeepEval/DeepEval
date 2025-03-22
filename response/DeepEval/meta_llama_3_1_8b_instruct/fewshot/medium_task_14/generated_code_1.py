import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn1 = BatchNormalization()(conv1)
    block1_output = bn1

    # Block 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    bn2 = BatchNormalization()(conv2)
    block2_output = bn2

    # Block 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    bn3 = BatchNormalization()(conv3)
    block3_output = bn3

    # Parallel Branch
    parallel_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Outputs
    adding_layer = Add()([block3_output, parallel_output])

    # Flatten and Dense Layers
    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model