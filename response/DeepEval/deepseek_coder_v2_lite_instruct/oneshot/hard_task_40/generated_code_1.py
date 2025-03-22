import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, Flatten, Dense, Reshape, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)
    
    # Flatten and concatenate the outputs
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])
    
    # Reshape the concatenated output into a 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 12 * 28 * 28))(concatenated)
    
    # Second block
    def second_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 followed by two 3x3 convolutions
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 followed by a single 3x3 convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: 1x1 convolution followed by average pooling
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(path4)
        
        # Concatenate the outputs of all paths
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        # Dropout to mitigate overfitting
        output_tensor = Dropout(0.5)(output_tensor)
        
        return output_tensor
    
    second_block_output = second_block(reshape_layer)
    
    # Flatten the output of the second block
    flatten_layer = Flatten()(second_block_output)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model