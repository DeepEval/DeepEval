import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer with the specified shape
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    # 1x1 convolution for dimensionality reduction
    main_path_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Two parallel convolutional layers for feature extraction
    main_path_2a = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_1)
    main_path_2b = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_1)
    
    # Concatenate the outputs of the parallel layers
    main_path_output = Concatenate()([main_path_2a, main_path_2b])
    
    # Branch path
    # 3x3 convolutional layer
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch path outputs using addition
    combined_output = Add()([main_path_output, branch_path])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined_output)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model