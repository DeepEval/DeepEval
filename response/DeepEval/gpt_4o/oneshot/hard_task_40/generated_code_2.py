import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape
from keras.layers import Conv2D, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First Block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    concat = Concatenate()([flat1, flat2, flat3])
    dense_reshape = Dense(units=784, activation='relu')(concat)
    reshaped = Reshape((7, 7, 16))(dense_reshape)

    # Second Block
    def multi_scale_block(input_tensor):
        # Path 1: 1x1 Conv
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path1 = Dropout(0.2)(path1)

        # Path 2: 1x1 Conv followed by two 3x3 Conv
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Dropout(0.2)(path2)

        # Path 3: 1x1 Conv followed by a 3x3 Conv
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Dropout(0.2)(path3)

        # Path 4: Average Pooling followed by 1x1 Conv
        path4 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
        path4 = Dropout(0.2)(path4)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = multi_scale_block(input_tensor=reshaped)

    # Final Classification Layers
    flatten_layer = Flatten()(block_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model