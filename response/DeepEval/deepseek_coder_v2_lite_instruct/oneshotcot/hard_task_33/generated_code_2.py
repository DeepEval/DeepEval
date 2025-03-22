import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define a block function
    def block(input_tensor):
        # Step 4.1: Add 1x1 convolutional layer
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        
        # Step 4.2: Add 3x3 depthwise separable convolutional layer
        path2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        
        # Step 4.3: Add another 1x1 convolutional layer
        path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        
        # Step 4.5: Add the block's input to form the output
        output_tensor = keras.layers.add([input_tensor, path3])
        
        return output_tensor
    
    # Apply the block to each branch
    branch1 = block(input_layer)
    branch2 = block(input_layer)
    branch3 = block(input_layer)
    
    # Concatenate the outputs of the three branches
    concat_layer = Concatenate()([branch1, branch2, branch3])
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(concat_layer)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model