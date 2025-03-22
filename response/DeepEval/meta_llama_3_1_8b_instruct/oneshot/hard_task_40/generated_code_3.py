import keras
from keras.layers import Input, AveragePooling2D, Concatenate, Conv2D, Dropout, Reshape, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # First block: three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(avg_pool1)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=4, padding='valid')(avg_pool2)

    # Concatenate the outputs of the pooling layers
    pooling_output = Concatenate()([avg_pool1, avg_pool2, avg_pool3])

    # Flatten the output
    flatten_layer = Flatten()(pooling_output)

    # Reshape the output into a 4-dimensional tensor
    reshaped_layer = Reshape((1, 1, 128))(flatten_layer)

    # Second block: four parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)

        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)

        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(path4)

        # Concatenate the outputs of the paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        # Apply dropout
        output_tensor = Dropout(0.2)(output_tensor)

        return output_tensor

    block_output = block(reshaped_layer)
    bath_norm = block_output
    flatten_layer2 = Flatten()(bath_norm)

    # Output layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model