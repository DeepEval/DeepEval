import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Reshape, Concatenate, Conv2D, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        # Path 1: MaxPooling with pool size 1x1
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(rate=0.5)(flatten1)

        # Path 2: MaxPooling with pool size 2x2
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(rate=0.5)(flatten2)

        # Path 3: MaxPooling with pool size 4x4
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(rate=0.5)(flatten3)

        # Concatenate all paths
        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    def block_2(input_tensor):
        # Path 1: 1x1 Convolution
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)

        # Path 2: 1x1 Convolution followed by 1x7 and 7x1 convolutions
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(conv2_1)
        conv2_3 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(conv2_2)

        # Path 3: 1x1 Convolution followed by alternating 7x1 and 1x7 convolutions
        conv3_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3_2 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(conv3_1)
        conv3_3 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(conv3_2)
        conv3_4 = Conv2D(filters=64, kernel_size=(7, 1), padding='same', activation='relu')(conv3_3)
        conv3_5 = Conv2D(filters=64, kernel_size=(1, 7), padding='same', activation='relu')(conv3_4)

        # Path 4: Average Pooling followed by 1x1 Convolution
        avg_pool = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        conv4 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(avg_pool)

        # Concatenate all paths
        output_tensor = Concatenate()([conv1, conv2_3, conv3_5, conv4])
        return output_tensor

    # Build Block 1
    block1_output = block_1(input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(4, 4, 8))(dense)  # Adjust target shape as needed

    # Build Block 2
    block2_output = block_2(reshaped)

    # Final fully connected layers for classification
    flatten = Flatten()(block2_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model