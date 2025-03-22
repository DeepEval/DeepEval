import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Step 2: Main path - 1x1 convolution for dimensionality reduction
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Parallel convolutional layers for feature extraction
    conv_1x1_parallel = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    conv_3x3_parallel = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)
    
    # Step 4: Concatenate the outputs of the two parallel layers
    main_path_output = Concatenate()([conv_1x1_parallel, conv_3x3_parallel])
    
    # Step 5: Branch path - 3x3 convolution to match dimensions
    branch_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 6: Combine main path and branch path using addition
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Step 7: Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Step 8: Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Step 9: Output layer for classification probabilities
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model