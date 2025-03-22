import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block with multiple scale convolutions and pooling
    def block_1(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)

    # Flatten the output of the first block
    flatten_layer = Flatten()(block1_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model