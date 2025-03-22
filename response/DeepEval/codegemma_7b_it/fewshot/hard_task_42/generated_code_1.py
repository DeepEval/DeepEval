import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense, Reshape

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Parallel Max Pooling Paths
    def block_1(input_tensor):
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        maxpool1 = Dropout(0.25)(maxpool1)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        maxpool2 = Dropout(0.25)(maxpool2)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        maxpool3 = Dropout(0.25)(maxpool3)
        output_tensor = Concatenate()([maxpool1, maxpool2, maxpool3])
        return output_tensor

    # Block 2: Multi-Scale Feature Extraction Paths
    def block_2(input_tensor):
        # Path 1: Single 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 Convolution + 1x7 Convolution + 7x1 Convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)

        # Path 3: 1x1 Convolution + Alternating 7x1 and 1x7 Convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)

        # Path 4: Average Pooling + 1x1 Convolution
        path4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        # Concatenate Path Outputs
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    # Combine Blocks
    block1_output = block_1(input_tensor=input_layer)
    reshaped = Reshape(target_shape=(4, 4, 96))(block1_output)
    block2_output = block_2(input_tensor=reshaped)

    # Classification Layers
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model