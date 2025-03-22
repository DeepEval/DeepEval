import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Define main path with Lambda layer to split input
    def main_path(input_tensor):
        # Split the input tensor into three parts along the last dimension
        split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # First group remains unchanged
        group1 = split_layers[0]
        
        # Second group undergoes a 3x3 convolution
        group2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
        
        # Combine second group with third group and apply another 3x3 convolution
        combined_group2_3 = Concatenate()([group2, split_layers[2]])
        group3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(combined_group2_3)
        
        # Concatenate the outputs of all three groups
        output_tensor = Concatenate()([group1, group2, group3])
        return output_tensor

    # Step 3: Define branch path with a 1x1 convolution
    def branch_path(input_tensor):
        output_tensor = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        return output_tensor
    
    # Step 4: Apply paths to the input
    main_output = main_path(input_layer)
    branch_output = branch_path(input_layer)
    
    # Step 5: Fuse main and branch paths by addition
    fused_output = Add()([main_output, branch_output])
    
    # Step 6: Flatten the fused output
    flatten_layer = Flatten()(fused_output)
    
    # Step 7: Add a fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model