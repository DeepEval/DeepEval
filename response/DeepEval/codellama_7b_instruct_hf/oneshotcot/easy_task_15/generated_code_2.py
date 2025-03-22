import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    
    # Specialized block
    def specialized_block(input_tensor):
        # 3x3 convolutional layer
        path1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 1x1 convolutional layer
        path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Average pooling layer
        path3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        
        # Dropout layer
        path4 = Dropout(rate=0.2)(input_tensor)
        
        # Concatenate outputs
        output_tensor = Concatenate()([path1, path2, path3, path4])
        
        return output_tensor
    
    # Apply specialized block
    block_output = specialized_block(max_pooling)
    
    # Batch normalization
    bath_norm = BatchNormalization()(block_output)
    
    # Flatten layer
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model