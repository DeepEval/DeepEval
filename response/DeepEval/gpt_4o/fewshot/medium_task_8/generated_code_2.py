import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(input_tensor):
        # Split input into three groups
        groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # First group remains unchanged
        group1 = groups[0]
        
        # Second group undergoes a 3x3 convolution
        group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(groups[1])
        
        # Combine second group with the third group, and pass through an additional 3x3 convolution
        combined_group23 = Concatenate()([group2, groups[2]])
        group23_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group23)
        
        # Concatenate outputs of all three groups
        output = Concatenate()([group1, group2, group23_conv])
        return output

    # Branch path
    def branch_path(input_tensor):
        # Process the input with a 1x1 convolution
        output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        return output

    # Apply main path
    main_output = main_path(input_layer)
    
    # Apply branch path
    branch_output = branch_path(input_layer)
    
    # Fuse outputs of main and branch paths through addition
    fused_output = Add()([main_output, branch_output])
    
    # Flatten the fused output
    flattened = Flatten()(fused_output)
    
    # Final fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model