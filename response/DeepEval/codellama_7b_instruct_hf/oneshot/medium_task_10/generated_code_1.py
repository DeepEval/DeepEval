import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Adjust input feature dimensionality to 16 using a convolutional layer
    conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Basic block
    def basic_block(input_tensor):
        conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        output_tensor = Concatenate()([conv1, conv2, conv3])

        return output_tensor

    # First level
    basic_block_output = basic_block(input_tensor=conv)

    # Second level
    residual_block1 = basic_block(input_tensor=basic_block_output)
    residual_block2 = basic_block(input_tensor=conv)
    output_tensor = Concatenate()([residual_block1, residual_block2])

    # Third level
    global_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    output_tensor = Concatenate()([output_tensor, global_branch])

    # Final layer
    flatten_layer = Flatten()(output_tensor)
    dense = Dense(units=10, activation='softmax')(flatten_layer)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model