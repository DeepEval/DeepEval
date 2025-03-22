import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, Conv2D
from keras.models import Model

def dl_model():     
    input_layer = Input(shape=(28, 28, 1))

    # First block with average pooling
    avg_pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening and concatenating the outputs
    flat_1 = Flatten()(avg_pool_1)
    flat_2 = Flatten()(avg_pool_2)
    flat_3 = Flatten()(avg_pool_3)
    concat_flat = Concatenate()([flat_1, flat_2, flat_3])
    
    # Fully connected layer
    fc_layer = Dense(units=128, activation='relu')(concat_flat)
    
    # Reshaping to prepare for second block
    reshaped = Reshape(target_shape=(1, 1, 128))(fc_layer)

    # Second block with parallel paths
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.3)(path1)

        # Path 2: Two stacked 3x3 convolutions after a 1x1 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Dropout(0.3)(path2)

        # Path 3: Single 3x3 convolution after a 1x1 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path3 = Dropout(0.3)(path3)

        # Path 4: Average pooling + 1x1 convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.3)(path4)

        # Concatenating the outputs from all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor

    block_output = second_block(input_tensor=reshaped)

    # Final fully connected layers for classification
    flatten_output = Flatten()(block_output)
    dense1 = Dense(units=64, activation='relu')(flatten_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model