import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    inputs = layers.Input(shape=input_shape)

    # Split the input into three groups along the channel dimension
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define the main path with multi-scale feature extraction
    main_path_outputs = []
    kernel_sizes = [1, 3, 5]
    
    for kernel_size in kernel_sizes:
        x = layers.SeparableConv2D(32, kernel_size=(kernel_size, kernel_size), padding='same', activation='relu')(splits[0])
        main_path_outputs.append(x)
    
    # Concatenate the outputs from the main path
    main_path = layers.Concatenate()(main_path_outputs)

    # Branch path with a 1x1 convolution to align the number of output channels
    branch_path = layers.Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Fuse the outputs from both paths
    fused_output = layers.Add()([main_path, branch_path])

    # Flatten the combined output
    flattened_output = layers.Flatten()(fused_output)

    # Fully connected layers for classification
    dense_1 = layers.Dense(128, activation='relu')(flattened_output)
    outputs = layers.Dense(10, activation='softmax')(dense_1)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Instantiate the model
model = dl_model()
model.summary()  # Display the model architecture