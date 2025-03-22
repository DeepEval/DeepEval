import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Main path
    # Split the input into three groups
    main_path_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # First group remains unchanged
    main_path_group1 = main_path_splits[0]
    
    # Second group undergoes feature extraction via a 3x3 convolutional layer
    main_path_group2 = Conv2D(32, (3, 3), activation='relu')(main_path_splits[1])
    
    # Combine the second and third groups before passing through an additional 3x3 convolution
    combined_group2_3 = Conv2D(32, (3, 3), activation='relu')(tf.concat([main_path_group2, main_path_splits[2]], axis=-1))
    
    # Concatenate the outputs of all three groups
    main_path_output = tf.concat([main_path_group1, combined_group2_3], axis=-1)

    # Branch path
    # Process the input with a 1x1 convolutional layer
    branch_path_output = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Fuse the outputs from both paths through addition
    fused_output = Add()([main_path_output, branch_path_output])

    # Flatten the combined output
    flattened_output = Flatten()(fused_output)

    # Pass the flattened output through a fully connected layer
    outputs = Dense(10, activation='softmax')(flattened_output)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Example usage:
# model = dl_model()
# model.summary()