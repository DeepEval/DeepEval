import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    
    input_layer = keras.Input(shape=(32, 32, 3))
    
    # Define a function to apply multi-scale feature extraction with separable convolutional layers
    def separable_convolution(input_tensor, kernel_size):
        return layers.SeparableConv2D(filters=32, kernel_size=kernel_size, padding='same')(input_tensor)
    
    # Split the input into three groups along the channel
    channel_split = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply multi-scale feature extraction with separable convolutional layers to each group
    conv1_group = separable_convolution(channel_split[0], kernel_size=(1, 1))
    conv3_group = separable_convolution(channel_split[1], kernel_size=(3, 3))
    conv5_group = separable_convolution(channel_split[2], kernel_size=(5, 5))
    
    # Concatenate the outputs from the three groups
    main_path_output = layers.Concatenate()([conv1_group, conv3_group, conv5_group])
    
    # Define a branch path that applies a 1x1 convolutional layer to the input
    branch_path_output = layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Fuse the outputs from both paths through addition
    fused_output = layers.Add()([main_path_output, branch_path_output])
    
    # Flatten the fused output into a one-dimensional vector
    flatten_layer = layers.Flatten()(fused_output)
    
    # Apply two fully connected layers for a 10-class classification task
    dense1 = layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = layers.Dense(units=10, activation='softmax')(dense1)
    
    # Create a Keras model from the input layer and the output layer
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model