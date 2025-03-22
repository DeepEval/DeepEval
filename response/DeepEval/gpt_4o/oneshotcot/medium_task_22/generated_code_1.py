import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.layers import BatchNormalization

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Step 4: Define the multi-branch block
    def multi_branch_block(input_tensor):
        # Branch 1: 3x3 Convolutions
        branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        
        # Branch 2: 1x1 Convolution followed by two 3x3 Convolutions
        branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        branch2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
        
        # Branch 3: Max Pooling
        branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        
        # Step 4.5: Concatenate the outputs of the branches
        concatenated = Concatenate()([branch1, branch2, branch3])
        
        return concatenated
    
    # Pass the input through the multi-branch block
    multi_branch_output = multi_branch_block(input_layer)
    
    # Step 6: Add flatten layer
    flatten_layer = Flatten()(multi_branch_output)
    
    # Step 7: Add dense layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Step 8: Add dense layer
    output_layer = Dense(units=10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model