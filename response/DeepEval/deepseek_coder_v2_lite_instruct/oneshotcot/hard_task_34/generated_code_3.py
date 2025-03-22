import keras
from keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    def main_path_block(input_tensor):
        # Step 2: Add separable convolutional layer with ReLU activation
        conv = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Step 3: Add batch normalization
        conv = BatchNormalization()(conv)
        return conv
    
    # Apply the main path block three times
    x = main_path_block(input_layer)
    x = main_path_block(x)
    x = main_path_block(x)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Concatenate the main path and branch path along the channel dimension
    combined = Concatenate(axis=-1)([x, branch_path])
    
    # Step 5: Add flatten layer
    flatten_layer = Flatten()(combined)
    
    # Step 7: Add dense layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model