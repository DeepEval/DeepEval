import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Basic block with convolution, batch normalization, and ReLU activation
    def block(input_tensor):
        batch_norm = BatchNormalization()(input_tensor)
        relu = keras.activations.relu(batch_norm)
        return relu

    # Main path of the basic block
    main_path_output = block(conv1)
    
    # Branch path of the basic block
    branch_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Concatenate the outputs of the main path and branch path
    concat = Concatenate()([main_path_output, branch_output])
    
    # Additional convolutional layer for the global branch
    global_branch_conv = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    
    # Add the outputs of the global branch to the second level residual structure
    second_level_output = Add()([concat, global_branch_conv])
    
    # Second level residual block
    def residual_block(input_tensor):
        block_output = block(input_tensor)
        return keras.activations.relu(block_output + input_tensor)
    
    second_level_output = residual_block(second_level_output)
    
    # Third level residual block
    third_level_output = residual_block(second_level_output)
    
    # Global average pooling and fully connected layer for classification
    average_pooling = MaxPooling2D(pool_size=(4, 4), strides=1)(third_level_output)
    flatten = Flatten()(average_pooling)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])