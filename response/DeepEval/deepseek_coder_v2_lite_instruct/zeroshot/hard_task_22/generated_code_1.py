import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Add, Flatten, Dense, Lambda

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Split the input into three groups along the channel
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    
    # Multi-scale feature extraction with separable convolutional layers
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(split_layer[0])
    conv3x3 = Conv2D(32, (3, 3), padding='same', activation='relu')(split_layer[1])
    conv5x5 = Conv2D(32, (5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from the three groups
    main_path_output = Add()([conv1x1, conv3x3, conv5x5])
    
    # Branch path
    # Apply a 1x1 convolutional layer to the input to align the number of output channels
    branch_path_output = Conv2D(32, (1, 1), activation='relu')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fused_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output into a one-dimensional vector
    flattened_output = Flatten()(fused_output)
    
    # Pass through two fully connected layers for a 10-class classification task
    dense1 = Dense(128, activation='relu')(flattened_output)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Example usage
model = dl_model()
model.summary()