import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    split_ratio = 3
    split_axis = -1
    splits = tf.split(inputs, num_or_size_splits=split_ratio, axis=split_axis)
    
    # Apply 1x1 convolutions to each group independently
    kernel_size = (1, 1)
    conv_kernels = [len(inputs.shape) - 1] // split_ratio
    conv_layers = []
    for i in range(split_ratio):
        conv_layers.append(Conv2D(conv_kernels[i], kernel_size, padding='same', activation='relu')(splits[i]))
    
    # Downsample each group via average pooling
    pool_size = (2, 2)
    pool_strides = (2, 2)
    pool_layers = [AveragePooling2D(pool_size, strides=pool_strides)(conv_layer) for conv_layer in conv_layers]
    
    # Concatenate the feature maps along the channel dimension
    concatenated = tf.concat(pool_layers, axis=split_axis)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()