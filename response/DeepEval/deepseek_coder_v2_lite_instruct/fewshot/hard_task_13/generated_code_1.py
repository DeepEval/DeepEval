import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, Reshape, Multiply

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block_1(input_tensor):
        # 1x1 convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 convolution
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 5x5 convolution
        conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 max pooling
        pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        # Concatenate the outputs of the four branches
        output_tensor = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])
        return output_tensor

    def block_2(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers
        dense1 = Dense(units=64, activation='relu')(gap)
        dense2 = Dense(units=10, activation='softmax')(dense1)
        return dense2

    # First block
    block1_output = block_1(input_tensor=input_layer)
    # Second block
    final_output = block_2(input_tensor=block1_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=final_output)

    return model