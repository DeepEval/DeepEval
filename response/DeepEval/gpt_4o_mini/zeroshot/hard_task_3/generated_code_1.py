import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Split the input into 3 groups along the channel dimension
    split_inputs = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Create the main pathway for each split
    feature_layers = []
    for split_input in split_inputs:
        x = layers.Conv2D(32, (1, 1), activation='relu')(split_input)  # 1x1 convolution
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)  # 3x3 convolution
        x = layers.Dropout(0.5)(x)  # Dropout for feature selection
        feature_layers.append(x)

    # Concatenate the outputs from the main pathway
    main_pathway = layers.Concatenate()(feature_layers)

    # Create the branch pathway with 1x1 convolution
    branch_pathway = layers.Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Combine both pathways using an addition operation
    combined = layers.Add()([main_pathway, branch_pathway])

    # Fully connected layer for classification
    x = layers.GlobalAveragePooling2D()(combined)  # Global average pooling
    output_layer = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Construct the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
if __name__ == "__main__":
    model = dl_model()
    model.summary()  # Print the model summary to verify architecture