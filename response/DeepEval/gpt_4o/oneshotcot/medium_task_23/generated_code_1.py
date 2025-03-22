import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def multi_branch_block(input_tensor):
        # Path 1: Single 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 Convolution followed by 1x7 and 7x1 Convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 Convolution followed by two sets of 1x7 and 7x1 Convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)

        # Path 4: Average Pooling followed by 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Apply multi-branch block
    block_output = multi_branch_block(input_tensor=input_layer)
    
    # Flatten the output
    flatten_layer = Flatten()(block_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model