# Import necessary packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    """
    A deep learning model for image classification using the CIFAR-10 dataset.
    The model splits the input image into three groups along the channel dimension
    using Lambda layer and applies different convolutional kernels: 1x1, 3x3, and 5x5,
    for multi-scale feature extraction. The outputs from these three groups are
    concatenated to fuse the features. Finally, the fused features are flattened
    into a one-dimensional vector and passed through two fully connected layers
    for classification.
    """
    
    # Define the input shape of the model
    input_shape = (32, 32, 3)
    
    # Create the model
    inputs = keras.Input(shape=input_shape)
    
    # Split the input image into three groups along the channel dimension
    split_output = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Apply different convolutional kernels: 1x1, 3x3, and 5x5
    conv1x1 = layers.Conv2D(32, (1, 1), activation='relu')(split_output[0])
    conv3x3 = layers.Conv2D(32, (3, 3), activation='relu')(split_output[1])
    conv5x5 = layers.Conv2D(32, (5, 5), activation='relu')(split_output[2])
    
    # Concatenate the outputs from the three groups
    concatenated = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Flatten the concatenated features into a one-dimensional vector
    flattened = layers.Flatten()(concatenated)
    
    # Apply two fully connected layers for classification
    fc1 = layers.Dense(128, activation='relu')(flattened)
    outputs = layers.Dense(10, activation='softmax')(fc1)
    
    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model