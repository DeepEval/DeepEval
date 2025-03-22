import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    # Input layer for the model
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main pathway: 1x1 convolution
    main_path_1x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Parallel branch: 1x1, 1x3, and 3x1 convolutions
    parallel_path_1x1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    parallel_path_1x3 = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(input_layer)
    parallel_path_3x1 = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(input_layer)

    # Concatenate outputs from the parallel paths
    concatenated = layers.concatenate([parallel_path_1x1, parallel_path_1x3, parallel_path_3x1], axis=-1)

    # Combine the main pathway and the concatenated features
    combined = layers.add([main_path_1x1, concatenated])

    # 1x1 convolution to produce the output of the same dimensions as the input channels
    output_layer = layers.Conv2D(3, (1, 1), padding='same', activation='relu')(combined)

    # Flatten the output before feeding into fully connected layers
    flattened = layers.Flatten()(output_layer)

    # Fully connected layers for classification
    dense_1 = layers.Dense(128, activation='relu')(flattened)
    dense_2 = layers.Dense(10, activation='softmax')(dense_1)  # CIFAR-10 has 10 classes

    # Create the model
    model = models.Model(inputs=input_layer, outputs=dense_2)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of creating the model
model = dl_model()
model.summary()  # This will print the model architecture