import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Path 1: 1x1 MaxPooling
        path1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path1 = Flatten()(path1)
        
        # Path 2: 2x2 MaxPooling
        path2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        path2 = Flatten()(path2)
        
        # Path 3: 4x4 MaxPooling
        path3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        path3 = Flatten()(path3)
        
        # Concatenate the outputs of the three paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        # Dropout for regularization
        output_tensor = Dropout(0.5)(output_tensor)
        
        return output_tensor
    
    block1_output = block1(input_tensor=input_layer)
    
    # Fully connected layer and reshape to transform the output of Block 1 into a 4-dimensional tensor
    reshape_layer = Reshape((1, 1, 3))(block1_output)
    
    # Block 2
    def block2(input_tensor):
        # Path 1: 1x1 Convolution
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x1 -> 1x7 -> 7x1 Convolution
        path2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path2)
        path2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Path 3: 1x1 -> 7x1 -> 1x7 Convolution
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(path3)
        path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(path3)
        
        # Path 4: Average Pooling followed by 1x1 Convolution
        path4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        path4 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path4)
        
        # Concatenate the outputs of the four paths along the channel dimension
        output_tensor = Concatenate(axis=-1)([path1, path2, path3, path4])
        
        return output_tensor
    
    block2_output = block2(input_tensor=reshape_layer)
    
    # Batch Normalization
    batch_norm = BatchNormalization()(block2_output)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Two fully connected layers for final classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model