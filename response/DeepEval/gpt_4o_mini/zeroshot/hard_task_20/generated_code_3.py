import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Splitting the input into three groups using Lambda layer
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Main path processing with different kernel sizes
    conv1 = layers.Conv2D(32, (1, 1), activation='relu')(split_inputs[0])
    conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(split_inputs[1])
    conv3 = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(split_inputs[2])

    # Concatenating the outputs of the main path
    main_output = layers.Concatenate()([conv1, conv2, conv3])
    
    # Branch path with a 1x1 convolution to align channels
    branch_output = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Combining main and branch outputs through addition
    combined_output = layers.Add()([main_output, branch_output])

    # Further processing before classification
    combined_output = layers.Flatten()(combined_output)
    combined_output = layers.Dense(128, activation='relu')(combined_output)
    combined_output = layers.Dense(64, activation='relu')(combined_output)
    
    # Output layer with softmax activation for classification
    output_layer = layers.Dense(10, activation='softmax')(combined_output)  # 10 classes for CIFAR-10

    # Constructing the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()

# Optionally print the model summary
model.summary()