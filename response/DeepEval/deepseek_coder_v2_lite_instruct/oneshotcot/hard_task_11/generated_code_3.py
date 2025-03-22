import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Add convolutional layer (main pathway)
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add maxpooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv_main)
    
    # Step 4: Define a block
    def block(input_tensor):
        # 1x1 convolution path
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 1x3 convolution path
        path2 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # 3x1 convolution path
        path3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Concatenate the outputs of these paths
        output_tensor = Concatenate()([path1, path2, path3])
        
        # Another 1x1 convolution
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
        
        return output_tensor
    
    # Apply the block to the maxpooling output
    block_output = block(input_tensor=max_pooling)
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(block_output)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    dense2 = Dense(units=128, activation='relu')(dense1)
    
    # Step 9: Add dense layer for output
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model