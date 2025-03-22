import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Step 2: Main path - Dimensionality reduction with 1x1 convolution
    main_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Main path - Parallel convolutional layers
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    
    # Step 4: Concatenate outputs of parallel paths
    main_path_output = Concatenate()([path1, path2])
    
    # Step 5: Branch path - 3x3 convolutional layer
    branch_path_output = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 6: Combine outputs from main path and branch path using addition
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Step 7: Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Step 8: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Step 9: Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model