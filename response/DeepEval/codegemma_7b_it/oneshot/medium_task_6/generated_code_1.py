import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Initial convolution
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel blocks
    def block(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor

    block_output1 = block(input_tensor=conv)
    block_output2 = block(input_tensor=block_output1)
    block_output3 = block(input_tensor=block_output2)

    # Adding outputs
    added_output = keras.layers.Add()([conv, block_output1, block_output2, block_output3])
    bath_norm = BatchNormalization()(added_output)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model