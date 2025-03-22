import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Pooling and concatenation
    def block_1(input_tensor):
        avgpool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(avgpool1)
        avgpool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(avgpool2)
        avgpool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(avgpool3)
        output_tensor = Concatenate()([flatten1, flatten2, flatten3])
        return output_tensor

    # Second block: Multi-scale feature extraction
    def block_2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: 1x1 convolution followed by two stacked 3x3 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: 1x1 convolution followed by a 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average pooling followed by a 1x1 convolution
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenate paths
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        return output_tensor

    # Connecting the blocks
    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(7, 7, 4))(dense)  # Assuming appropriate reshaping
    block2_output = block_2(input_tensor=reshaped)

    # Final fully connected layers for classification
    flatten = Flatten()(block2_output)
    fc1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model