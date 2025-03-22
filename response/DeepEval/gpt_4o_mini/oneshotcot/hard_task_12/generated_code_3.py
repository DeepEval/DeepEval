import keras
from keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))
    
    # Main path: 1x1 convolutional layer for dimensionality reduction
    main_path = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Feature extraction via two parallel convolutions
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path)
    conv_3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    
    # Concatenate the outputs from the parallel convolutions
    concatenated = Concatenate()([conv_1x1, conv_3x3])
    
    # Branch path: 3x3 convolutional layer
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Ensure the output dimensions match between the main and branch paths
    # Using a convolutional layer to adjust dimensions if needed
    branch_adjusted = Conv2D(filters=64, kernel_size=(1, 1), padding='same')(branch_path)
    
    # Combine the main and branch paths using addition
    combined = Add()([concatenated, branch_adjusted])
    
    # Flatten the combined output
    flatten_layer = Flatten()(combined)
    
    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model