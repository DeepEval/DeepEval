import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Custom block
    def custom_block(input_tensor):
        # Path 1: 1x1 convolution
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Path 2: 1x3 convolution followed by 3x1 convolution
        path2 = Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Concatenate outputs from both paths
        combined = Concatenate()([path1, path2])
        
        # Final 1x1 convolution
        output_tensor = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(combined)
        
        return output_tensor
    
    block_output = custom_block(pool1)
    batch_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Direct connection from input to output for fusion
    direct_connection = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    direct_connection = BatchNormalization()(direct_connection)
    direct_connection = Flatten()(direct_connection)
    
    # Add the direct connection to the main pathway
    final_output = Add()([flatten_layer, direct_connection])
    
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model

# Create the model
model = dl_model()
model.summary()