import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path
    # Step 1: Dimensionality reduction with 1x1 convolution
    main_path_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 2: Feature extraction with parallel convolutions
    main_path_conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path_1x1)
    main_path_conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path_1x1)
    
    # Step 3: Concatenate the outputs
    main_path_concat = Concatenate()([main_path_conv1, main_path_conv3])
    
    # Branch path
    # 3x3 convolution to match main path's output dimensions
    branch_path = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch paths using addition
    combined_output = Add()([main_path_concat, branch_path])
    
    # Final layers for classification
    flatten_layer = Flatten()(combined_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model