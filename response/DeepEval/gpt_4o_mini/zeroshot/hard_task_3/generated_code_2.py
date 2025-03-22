import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    inputs = layers.Input(shape=input_shape)

    # Split the input along the channel dimension into three groups
    splits = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)

    # Define the main pathway
    def main_pathway(branch_input):
        x = layers.Conv2D(32, (1, 1), activation='relu')(branch_input)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.Dropout(0.5)(x)
        return x

    # Process each split through the main pathway
    branch_outputs = [main_pathway(split) for split in splits]

    # Concatenate the outputs from the three branches
    concatenated = layers.concatenate(branch_outputs, axis=-1)

    # Branch pathway
    branch_pathway = layers.Conv2D(192, (1, 1), activation='relu')(inputs)

    # Combine the outputs of the main and branch pathways
    combined = layers.add([concatenated, branch_pathway])

    # Fully connected layer for classification
    x = layers.GlobalAveragePooling2D()(combined)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model