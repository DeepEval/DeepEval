import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Dense, Flatten
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # The first group remains unchanged
    group1 = split_layer[0]
    
    # The second group undergoes feature extraction via a 3x3 convolutional layer
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    
    # Combine the second and third groups before passing through an additional 3x3 convolution
    combined_groups = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate(axis=-1)([group2, split_layer[2]]))
    
    # The outputs of all three groups are concatenated to form the output of the main path
    main_path_output = Concatenate(axis=-1)([group1, combined_groups, split_layer[2]])
    
    # Branch path
    # Process the input with a 1x1 convolutional layer
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fused_output = tf.add(main_path_output, branch_path_output)
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Pass the flattened output through a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model