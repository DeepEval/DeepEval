import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 RGB
    inputs = layers.Input(shape=input_shape)
    
    # Split the input along the channel dimension
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    # Main path with multi-scale feature extraction
    conv1x1 = layers.SeparableConv2D(32, (1, 1), padding='same', activation='relu')(splits[0])
    conv3x3 = layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu')(splits[1])
    conv5x5 = layers.SeparableConv2D(32, (5, 5), padding='same', activation='relu')(splits[2])
    
    # Concatenate the outputs from the three convolutions
    main_path_output = layers.Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch path with a 1x1 convolution
    branch_path_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    
    # Fuse the outputs from both paths
    combined_output = layers.Add()([main_path_output, branch_path_output])
    
    # Flatten the output
    flattened_output = layers.Flatten()(combined_output)
    
    # Fully connected layers
    dense1 = layers.Dense(128, activation='relu')(flattened_output)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    
    # Output layer for classification into 10 classes
    outputs = layers.Dense(10, activation='softmax')(dense2)

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model