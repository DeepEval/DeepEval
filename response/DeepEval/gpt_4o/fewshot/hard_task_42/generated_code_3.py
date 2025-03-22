import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Concatenate, Dense, Reshape, Conv2D, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1 with parallel max pooling paths
    def block_1(input_tensor):
        pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(pool1)
        dropout1 = Dropout(rate=0.3)(flatten1)

        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(pool2)
        dropout2 = Dropout(rate=0.3)(flatten2)

        pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(pool3)
        dropout3 = Dropout(rate=0.3)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])
        return output_tensor

    # Block 2 with parallel convolution and pooling paths
    def block_2(input_tensor):
        # Path 1
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)

        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block1_output = block_1(input_tensor=input_layer)
    dense = Dense(units=128, activation='relu')(block1_output)
    reshaped = Reshape(target_shape=(8, 8, 2))(dense)  # Adjust the shape as necessary for block 2
    block2_output = block_2(input_tensor=reshaped)

    flatten = Flatten()(block2_output)
    dense2 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model