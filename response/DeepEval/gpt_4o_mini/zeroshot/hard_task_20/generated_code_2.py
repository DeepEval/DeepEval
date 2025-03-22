import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Main path
    # Split the input into three groups
    split_inputs = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Feature extraction with different kernel sizes
    conv1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(split_inputs[0])
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(split_inputs[1])
    conv3 = layers.Conv2D(32, (5, 5), padding='same', activation='relu')(split_inputs[2])

    # Concatenate the outputs of the three convolutions
    main_output = layers.concatenate([conv1, conv2, conv3], axis=-1)

    # Branch path
    branch_output = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Combine the outputs of the main and branch paths
    fused_output = layers.add([main_output, branch_output])

    # Global Average Pooling to reduce dimensions
    pooled_output = layers.GlobalAveragePooling2D()(fused_output)

    # Fully connected layers for classification
    dense1 = layers.Dense(128, activation='relu')(pooled_output)
    outputs = layers.Dense(10, activation='softmax')(dense1)  # CIFAR-10 has 10 classes

    # Construct the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

# Example of how to create the model
model = dl_model()
model.summary()  # Prints the model summary