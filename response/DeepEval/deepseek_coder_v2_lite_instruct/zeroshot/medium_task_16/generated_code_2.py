import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate

def dl_model():
    input_shape = (32, 32, 3)
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Split the input into three groups along the channel dimension
    splits = tf.split(inputs, num_or_size_splits=3, axis=-1)
    
    # Apply 1x1 convolutions to each group
    kernels = [input_shape[-1] // 3] * 3
    conv_groups = []
    for split, num_kernels in zip(splits, kernels):
        conv = Conv2D(num_kernels, kernel_size=(1, 1), activation='relu')(split)
        conv_groups.append(conv)
    
    # Downsample each group via average pooling
    pool_groups = []
    for conv_group in conv_groups:
        pool = AveragePooling2D(pool_size=(8, 8), strides=(8, 8))(conv_group)
        pool_groups.append(pool)
    
    # Concatenate the three groups of feature maps along the channel dimension
    concatenated = Concatenate(axis=-1)(pool_groups)
    
    # Flatten the concatenated feature maps into a one-dimensional vector
    flattened = Flatten()(concatenated)
    
    # Pass through two fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    outputs = Dense(10, activation='softmax')(fc1)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()