import keras
from keras.layers import Input, Conv2D, Dropout, BatchNormalization, Flatten, Add, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path(input_tensor):
        # Conv2D with dropout
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        conv1 = Dropout(0.2)(conv1)
        
        # Conv2D to restore channels
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(conv1)
        
        return conv2
    
    main_output = main_path(input_tensor=input_layer)
    
    # Branch path
    branch_output = input_tensor
    
    # Add the outputs from both paths
    combined_output = Add()([main_output, branch_output])
    
    # Flattening layer
    flatten_layer = Flatten()(combined_output)
    
    # Dense layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model