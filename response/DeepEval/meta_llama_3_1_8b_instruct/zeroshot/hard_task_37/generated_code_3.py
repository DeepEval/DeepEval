# Import necessary Keras and other packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Define the function for the deep learning model
def dl_model():
    """
    This function constructs a deep learning model for image classification using MNIST dataset.
    
    The model features two parallel branches, each consisting of a same block. The block contains three sequential convolutional layers,
    with the outputs of each layer directed to separate main paths. Additionally, a parallel branch connects the input through a convolutional layer.
    The outputs from all paths are combined using an addition operation. Subsequently, the outputs from the two blocks are concatenated.
    After passing through a flattening layer and a fully connected layer, the model produces the final classification results.
    """

    # Define the input shape for the model
    inputs = keras.Input(shape=(28, 28, 1))

    # Define the branch for the same block
    def same_block(x):
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x)
        return x

    # Define the branch for the parallel path
    parallel_path = layers.Conv2D(32, (3, 3), activation='relu')(inputs)

    # Define the block with two parallel branches
    block1 = same_block(inputs)
    block2 = same_block(inputs)

    # Combine the outputs from all paths using addition
    x = layers.Add()([block1, block2, parallel_path])

    # Define the output from the block
    x = layers.Add()([same_block(block1), same_block(block2), parallel_path])

    # Concatenate the outputs from the two blocks
    x = layers.Concatenate()([x, x])

    # Pass through a flattening layer
    x = layers.Flatten()(x)

    # Pass through a fully connected layer
    x = layers.Dense(64, activation='relu')(x)

    # Output layer for classification
    outputs = layers.Dense(10, activation='softmax')(x)

    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create the model
model = dl_model()
model.summary()