import keras
from keras.layers import Input, Conv2D, ReLU, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path_block(input_tensor):
        # First block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        x = BatchNormalization()(x)
        
        # Second block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # Third block
        x = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        return x
    
    # Apply the main path block three times
    main_path_output = main_path_block(input_layer)
    
    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path_output)
    
    # Concatenate the main path and branch path outputs along the channel dimension
    fused_output = Concatenate(axis=-1)([main_path_output, branch_path_output])
    
    # Flatten the fused output
    flattened_output = Flatten()(fused_output)
    
    # Pass the flattened output through a fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model