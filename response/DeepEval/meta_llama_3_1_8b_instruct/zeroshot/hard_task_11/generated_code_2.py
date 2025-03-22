# Import necessary packages
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers

def dl_model():
    # Input layer (32x32x3 images)
    input_layer = keras.Input(shape=(32, 32, 3))

    # Main pathway (1x1 convolution)
    main_pathway = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
    main_pathway = layers.Conv2D(64, (1, 1), padding='same')(main_pathway)

    # Parallel branch (1x1, 1x3, 3x1 convolutions)
    parallel_branch = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
    parallel_branch = layers.Conv2D(64, (1, 3), padding='same')(parallel_branch)
    parallel_branch = layers.Conv2D(64, (3, 1), padding='same')(parallel_branch)

    # Concatenate main pathway and parallel branch
    concatenated = layers.Concatenate()([main_pathway, parallel_branch])

    # Fusion with main pathway via additive operation
    fused = layers.Add()([concatenated, main_pathway])

    # Final convolution layer (same dimensions as channel of input)
    final_conv = layers.Conv2D(64, (1, 1), padding='same')(fused)

    # Flatten the output
    flattened = layers.Flatten()(final_conv)

    # Fully connected layers for classification
    fc1 = layers.Dense(64, activation='relu')(flattened)
    fc2 = layers.Dense(10, activation='softmax')(fc1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model