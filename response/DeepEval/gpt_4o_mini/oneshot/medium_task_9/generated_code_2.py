import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # Initial convolutional layer to reduce dimensionality to 16
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_layer)
    
    def basic_block(input_tensor):
        # Main path: Convolution -> BatchNorm -> ReLU
        main_path = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)

        # Branch: Directly connects to the input
        branch = input_tensor
        
        # Feature fusion by adding both paths
        output_tensor = Add()([main_path, branch])
        output_tensor = ReLU()(output_tensor)  # Apply ReLU after addition
        
        return output_tensor

    # First basic block
    block1_output = basic_block(initial_conv)
    
    # Second basic block
    block2_output = basic_block(block1_output)
    
    # Final feature extraction layer
    feature_extraction = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(block2_output)
    
    # Combine outputs of the last block and the feature extraction layer
    final_output = Add()([block2_output, feature_extraction])
    final_output = ReLU()(final_output)  # Apply ReLU after addition

    # Downsample the feature map
    average_pooling = AveragePooling2D(pool_size=(2, 2))(final_output)
    
    # Flatten the result and pass through a fully connected layer
    flatten_layer = Flatten()(average_pooling)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model