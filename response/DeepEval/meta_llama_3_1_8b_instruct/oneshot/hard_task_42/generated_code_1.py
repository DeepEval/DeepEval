import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Dropout
from keras.regularizers import l2

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    def block1(input_tensor):
        # Define three parallel paths with different max pooling scales
        path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        path2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        path3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_tensor)
        
        # Concatenate the outputs from each path
        concatenated = Concatenate()([path1, path2, path3])
        
        # Flatten the concatenated output into one-dimensional vectors
        flatten = Flatten()(concatenated)
        
        # Apply dropout regularization to reduce overfitting
        dropout1 = Dropout(0.2)(flatten)
        dropout2 = Dropout(0.2)(dropout1)
        
        return dropout2
    
    block_output1 = block1(input_layer)
    
    # Transform the output of block 1 into a 4-dimensional tensor format
    reshape = Reshape((64, 4))(block_output1)
    
    def block2(input_tensor):
        # Define four parallel paths with different convolution and pooling strategies
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(path2)
        path2 = Concatenate()([path2, input_tensor])
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(1, 7), strides=1, padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=64, kernel_size=(7, 1), strides=1, padding='same', activation='relu')(path3)
        path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(input_tensor)
        path4 = AveragePooling2D(pool_size=(3, 3), strides=1, padding='same')(input_tensor)
        
        # Concatenate the outputs from all paths along the channel dimension
        concatenated = Concatenate()([path1, path2, path3, path4])
        
        # Reshape the concatenated output back to a 2D tensor
        reshaped = Reshape((64 * 4,))(concatenated)
        
        return reshaped
    
    block_output2 = block2(reshape)
    
    # Apply a fully connected layer to extract higher-level features
    dense1 = Dense(units=128, activation='relu')(block_output2)
    
    # Apply another fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model