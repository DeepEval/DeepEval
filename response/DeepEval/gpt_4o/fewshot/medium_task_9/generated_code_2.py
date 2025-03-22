import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    def basic_block(input_tensor):
        # Main path
        conv_main = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        bn_main = BatchNormalization()(conv_main)
        relu_main = ReLU()(bn_main)
        
        # Branch path (identity)
        branch = input_tensor
        
        # Feature fusion by adding both paths
        output_tensor = Add()([relu_main, branch])
        
        return output_tensor
    
    # First basic block
    block1_output = basic_block(input_tensor=initial_conv)
    
    # Second basic block
    block2_output = basic_block(input_tensor=block1_output)
    
    # Branch path for additional feature extraction
    branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(initial_conv)
    
    # Combining outputs from both paths
    combined_output = Add()([block2_output, branch_conv])
    
    # Final processing layers
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(combined_output)
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model