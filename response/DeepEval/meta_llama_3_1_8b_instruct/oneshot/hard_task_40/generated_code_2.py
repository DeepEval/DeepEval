import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Reshape, BatchNormalization, Flatten, Dense, Dropout

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # First block: three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='same')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)

    # Concatenate and flatten the outputs
    concat_output = Concatenate()([avg_pool1, avg_pool2, avg_pool3])
    flatten_output = Flatten()(concat_output)

    # Reshape the output for the second block
    reshape_output = Reshape((1, 1, 49))(flatten_output)

    # Second block: four parallel paths with multi-scale features
    def block(input_tensor):

        # Path 1: 1x1 convolution
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.2)(conv1)

        # Path 2: 1x1 convolution + 3x3 convolution + 3x3 convolution
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
        path2 = Dropout(0.2)(conv4)

        # Path 3: 1x1 convolution + 3x3 convolution
        conv5 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv6 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
        path3 = Dropout(0.2)(conv6)

        # Path 4: average pooling + 1x1 convolution
        maxpool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        conv7 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(maxpool)
        path4 = Dropout(0.2)(conv7)

        # Concatenate the outputs along the channel dimension
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor

    # Apply the second block
    block_output = block(reshape_output)
    block_output = BatchNormalization()(block_output)
    flatten_layer = Flatten()(block_output)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model