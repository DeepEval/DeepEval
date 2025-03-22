import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Dropout, Conv2D, BatchNormalization
from keras.models import Model

def dl_model():
    # Input layer for MNIST dataset
    input_layer = Input(shape=(28, 28, 1))
    
    # First block with average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening the outputs of the average pooling layers
    flatten1 = Flatten()(avg_pool1)
    flatten2 = Flatten()(avg_pool2)
    flatten3 = Flatten()(avg_pool3)

    # Concatenating the flattened outputs
    concat_pool = Concatenate()([flatten1, flatten2, flatten3])
    
    # Fully connected layer
    fc1 = Dense(units=128, activation='relu')(concat_pool)
    
    # Reshaping for the second block
    reshape_layer = Reshape(target_shape=(-1, 1, 128))(fc1)

    # Second block with four parallel paths
    def block(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2: Stacked 3x3 Convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3: 3x3 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4: Average Pooling with 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenating the outputs of the four paths
        output_tensor = Concatenate()([path1, path2, path3, path4])

        return output_tensor
    
    # Creating the second block
    block_output = block(reshape_layer)

    # Flattening the output from the second block
    flatten_block = Flatten()(block_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=64, activation='relu')(flatten_block)
    dense1 = Dropout(0.5)(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Constructing the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model