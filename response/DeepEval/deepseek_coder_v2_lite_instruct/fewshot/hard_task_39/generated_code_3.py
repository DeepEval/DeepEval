import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten1 = Flatten()(maxpool1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    flatten2 = Flatten()(maxpool2)
    maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)
    flatten3 = Flatten()(maxpool3)
    block1_output = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer and reshape
    dense1 = Dense(units=64, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 4))(dense1)

    # Block 2
    def block_2_branch(input_tensor, kernel_size):
        return Conv2D(filters=64, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu')(input_tensor)

    conv1x1 = block_2_branch(reshaped, (1, 1))
    conv3x3 = block_2_branch(reshaped, (3, 3))
    conv5x5 = block_2_branch(reshaped, (5, 5))
    maxpool3x3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(reshaped)

    block2_output = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool3x3])

    # Flatten and fully connected layer
    flatten = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model