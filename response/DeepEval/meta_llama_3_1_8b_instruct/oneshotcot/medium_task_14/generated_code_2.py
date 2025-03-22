import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = BatchNormalization()(block1_output)

    block2_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block1_output)
    block2_output = BatchNormalization()(block2_output)

    block3_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block3_output = BatchNormalization()(block3_output)

    direct_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    direct_path = BatchNormalization()(direct_path)

    # Concatenate the outputs from the three blocks and the direct path
    output_tensor = Concatenate()([block1_output, block2_output, block3_output, direct_path])

    # Apply average pooling to reduce the spatial dimensions
    output_tensor = keras.layers.AveragePooling2D(pool_size=(2, 2))(output_tensor)

    # Flatten the output tensor
    flatten_layer = Flatten()(output_tensor)

    # Dense layer 1
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Dense layer 2
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model