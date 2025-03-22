import keras
from keras.layers import Input, AveragePooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D, Dropout, BatchNormalization
from keras.models import Model

def dl_model():
    
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: First block with three average pooling layers
    avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flattening the outputs and concatenating
    flattened1 = Flatten()(avg_pool1)
    flattened2 = Flatten()(avg_pool2)
    flattened3 = Flatten()(avg_pool3)
    
    concatenated = Concatenate()([flattened1, flattened2, flattened3])

    # Fully connected layer after concatenation
    dense_layer = Dense(units=128, activation='relu')(concatenated)
    
    # Reshape the output into a 4D tensor for the second block
    reshaped = Reshape((1, 1, 128))(dense_layer)

    # Step 3: Second block with four parallel paths
    def block(input_tensor):
        # Path 1 - 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path1 = Dropout(0.5)(path1)

        # Path 2 - Two 3x3 convolutions stacked after a 1x1 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path2 = Dropout(0.5)(path2)

        # Path 3 - Single 3x3 convolution following a 1x1 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(path3)
        path3 = Dropout(0.5)(path3)

        # Path 4 - Average pooling followed by a 1x1 convolution
        path4 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(path4)
        path4 = Dropout(0.5)(path4)

        # Concatenate all paths
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    block_output = block(input_tensor=reshaped)

    # Final steps: flatten and fully connected layers for classification
    flatten_block_output = Flatten()(block_output)
    dense1 = Dense(units=64, activation='relu')(flatten_block_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model