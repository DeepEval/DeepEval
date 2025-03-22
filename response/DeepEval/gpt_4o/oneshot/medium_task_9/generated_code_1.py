import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    initial_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define a basic block
    def basic_block(input_tensor):
        # Main path: convolution -> batch normalization -> ReLU
        main_path = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        main_path = BatchNormalization()(main_path)
        main_path = ReLU()(main_path)
        
        # Branch path: simple connection to input
        branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)

        # Fuse features by adding the paths
        output_tensor = Add()([main_path, branch_path])
        
        return output_tensor

    # Apply two consecutive basic blocks
    block1_output = basic_block(input_tensor=initial_conv)
    block2_output = basic_block(input_tensor=block1_output)
    
    # Output from the second basic block
    # Apply another convolutional layer on the branch path
    branch_conv = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(block2_output)
    
    # Add the outputs from both paths to enhance feature representation
    enhanced_output = Add()([block2_output, branch_conv])
    
    # Apply average pooling, flatten and fully connected layer
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(enhanced_output)
    flatten_layer = Flatten()(avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model