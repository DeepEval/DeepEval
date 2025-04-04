import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Dropout, Reshape, Concatenate

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Path 1
        path1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(0.5)(path1)

        # Path 2
        path2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3
        path3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(0.5)(path3)

        # Concatenate the paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)

    # Fully connected layer and reshape for block 2
    fc1 = Dense(units=128, activation='relu')(block1_output)
    reshaped_tensor = Reshape((8, 8, 2))(fc1)  # Adjust dimensions as needed based on the task description

    # Block 2
    def block2(input_tensor):
        # Branch 1: 1x1 convolution
        branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Branch 2: 1x1 convolution, 3x3 convolution
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

        # Branch 3: 1x1 convolution, 3x3 convolution, 3x3 convolution
        branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
        branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)

        # Branch 4: average pooling, 1x1 convolution
        branch4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        branch4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

        # Concatenate the branches
        output_tensor = Concatenate()([branch1, branch2, branch3, branch4])
        return output_tensor

    block2_output = block2(reshaped_tensor)

    # Final classification layers
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=64, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model