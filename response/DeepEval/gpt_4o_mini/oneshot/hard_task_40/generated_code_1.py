import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Dropout, Reshape, Conv2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block with three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening the outputs from average pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenating the flattened outputs
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(concatenated)
    reshape_layer = Reshape((1, 1, 128))(dense1)  # Reshaping into 4D tensor

    # Second block with four parallel paths
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: Two 3x3 convolutions after a 1x1 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: One 3x3 convolution after a 1x1 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average pooling followed by a 1x1 convolution
        path4 = AveragePooling2D(pool_size=(2, 2))(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenating the outputs from all paths
        concatenated_paths = Concatenate()([path1, path2, path3, path4])
        
        return concatenated_paths

    block_output = second_block(reshape_layer)

    # Flattening the output and creating final fully connected layers
    flatten_block_output = Flatten()(block_output)
    dense2 = Dense(units=64, activation='relu')(flatten_block_output)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model