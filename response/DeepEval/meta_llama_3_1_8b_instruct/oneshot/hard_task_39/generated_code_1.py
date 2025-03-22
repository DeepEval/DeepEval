import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Reshape, Dense, Flatten, Conv2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_input = input_layer
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(block1_input)
    pool1_flat = Flatten()(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(block1_input)
    pool2_flat = Flatten()(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(block1_input)
    pool3_flat = Flatten()(pool3)
    block1_output = Concatenate()([pool1_flat, pool2_flat, pool3_flat])

    # Convert Block 1 output to 4D tensor
    reshape_layer = Reshape((3 * 3 * 3,))(block1_output)

    # Block 2
    def block2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor

    block2_output = block2(reshape_layer)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(block2_output)

    # Flatten the output
    flatten_layer = Flatten()(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model