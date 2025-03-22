import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # First group remains unchanged
    first_group = split_layer[0]
    
    # Second group undergoes feature extraction via a 3x3 convolutional layer
    second_group = Conv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    
    # Combine second group output with third group
    combined_second_third = Concatenate(axis=-1)([second_group, split_layer[2]])
    
    # Additional 3x3 convolution on the combined second and third group
    combined_conv = Conv2D(32, (3, 3), padding='same', activation='relu')(combined_second_third)
    
    # Concatenate all three groups to form the output of the main path
    main_path_output = Concatenate(axis=-1)([first_group, second_group, combined_conv])
    
    # Branch path: 1x1 convolutional layer to process the input
    branch_path_output = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse main and branch path outputs through addition
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Final classification result through a fully connected layer
    final_output = Dense(10, activation='softmax')(flattened_output)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=final_output)
    
    return model