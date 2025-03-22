import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Split the input into three groups along the channel dimension
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Feature extraction through depthwise separable convolutions
    conv_1x1 = layers.SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(splits[0])
    conv_3x3 = layers.SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(splits[1])
    conv_5x5 = layers.SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(splits[2])
    
    # Concatenate the outputs from the main path
    main_path_output = layers.Concatenate()([conv_1x1, conv_3x3, conv_5x5])

    # Branch path
    branch_output = layers.Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(inputs)

    # Add the main path and branch path outputs
    combined_output = layers.Add()([main_path_output, branch_output])

    # Flatten the output
    flattened_output = layers.Flatten()(combined_output)

    # Fully connected layers for classification
    dense_1 = layers.Dense(128, activation='relu')(flattened_output)
    outputs = layers.Dense(10, activation='softmax')(dense_1)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example usage
# model = dl_model()
# model.summary()