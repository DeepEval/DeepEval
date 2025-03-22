import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Path 1: 1x1 max pooling
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        # Path 2: 2x2 max pooling
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        # Path 3: 4x4 max pooling
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        
        # Flatten and concatenate the outputs
        flatten1 = Flatten()(path1)
        flatten2 = Flatten()(path2)
        flatten3 = Flatten()(path3)
        concatenated = Concatenate()([flatten1, flatten2, flatten3])
        
        # Dropout regularization
        dropout = Dropout(0.5)(concatenated)
        
        return dropout
    
    block1_output = block1(input_layer)
    
    # Fully connected layer and reshape
    fc_layer = Dense(units=256, activation='relu')(block1_output)
    reshape_layer = Reshape((4, 4, 16))(fc_layer)  # Assuming the output needs to be a 4x4x16 tensor
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 followed by 1x7 convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 followed by alternating 7x1 and 1x7 convolutions
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: Average pooling followed by 1x1 convolution
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        
        # Concatenate along the channel dimension
        concatenated = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return concatenated
    
    block2_output = block2(reshape_layer)
    
    # Flatten the output of Block 2
    flatten_layer = Flatten()(block2_output)
    
    # Fully connected layers for final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model