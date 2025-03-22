import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Define a block for each parallel branch
    def block(input_tensor):
        # Step 4.1: Add convolutional layer as the first path
        path1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Step 4.2: Add convolutional layer as the second path
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Step 4.2: Add convolutional layer as the third path
        path3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Step 4.5: Add concatenate layer to merge the above paths
        output_tensor = Add()([path1, path2, path3])
        
        return output_tensor
    
    # Apply the block to the input tensor for both parallel branches
    block1_output = block(input_tensor=input_layer)
    block2_output = block(input_tensor=input_layer)
    
    # Step 4.5: Concatenate the outputs from the two blocks
    concatenated_output = Concatenate()([block1_output, block2_output])
    
    # Step 5: Add batch normalization layer
    batch_norm = BatchNormalization()(concatenated_output)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(batch_norm)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model