import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Split the input into three groups
    split_1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # First group remains unchanged
    main_path_1 = split_1[0]
    
    # Second group undergoes feature extraction via a 3x3 convolutional layer
    main_path_2 = Conv2D(64, (3, 3), activation='relu')(split_1[1])
    
    # Output of the second group is combined with the third group
    combined_2_3 = Add()([main_path_2, split_1[2]])
    
    # Additional 3x3 convolution
    main_path_3 = Conv2D(64, (3, 3), activation='relu')(combined_2_3)
    
    # Concatenate the outputs of all three groups
    main_path_output = tf.concat([main_path_1, main_path_2, main_path_3], axis=-1)
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Fuse outputs from both paths through addition
    fused_output = Add()([main_path_output, branch_path])
    
    # Flatten the combined output
    flattened_output = Flatten()(fused_output)
    
    # Pass through a fully connected layer
    fc_output = Dense(10, activation='softmax')(flattened_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=fc_output)
    
    return model

# Example usage
model = dl_model()
model.summary()