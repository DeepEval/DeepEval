import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    def block1(input_tensor):
        # Path 1: MaxPooling with 1x1
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        path1 = Dropout(0.5)(path1)

        # Path 2: MaxPooling with 2x2
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: MaxPooling with 4x4
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        path3 = Dropout(0.5)(path3)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor

    block1_output = block1(input_layer)

    # Fully connected layer and reshaping operation between block 1 and block 2
    fc_layer = Dense(units=256, activation='relu')(block1_output)
    reshape_layer = Reshape(target_shape=(4, 4, 16))(fc_layer)  # Adjust dimensions as necessary

    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 -> 1x7 -> 7x1 Convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 -> (7x1 -> 1x7) x2 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)

        # Path 4: Average Pooling + 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block2_output = block2(reshape_layer)

    # Final Fully Connected Layers for Classification
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model