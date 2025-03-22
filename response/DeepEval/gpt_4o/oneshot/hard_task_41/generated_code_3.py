import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dropout, Dense, Concatenate, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Path 1: Average Pooling with 1x1 window
        path1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(rate=0.5)(path1)

        # Path 2: Average Pooling with 2x2 window
        path2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(rate=0.5)(path2)

        # Path 3: Average Pooling with 4x4 window
        path3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(rate=0.5)(path3)

        # Concatenate paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    def block_2(input_tensor):
        # Reshape the input to 4D tensor (e.g., assuming reshape to 7x7x64 for further processing)
        reshaped = Reshape((7, 7, -1))(input_tensor)

        # Branch 1: 1x1 Convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)

        # Branch 2: 1x1 Convolution -> 3x3 Convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 Convolution -> 3x3 Convolution -> 3x3 Convolution
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshaped)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

        # Branch 4: Average Pooling -> 1x1 Convolution
        branch4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(reshaped)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    # Construct the model
    block1_output = block_1(input_layer)
    dense_transform = Dense(units=7 * 7 * 64, activation='relu')(block1_output)
    block2_input = Reshape((7, 7, 64))(dense_transform)
    block2_output = block_2(block2_input)
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model