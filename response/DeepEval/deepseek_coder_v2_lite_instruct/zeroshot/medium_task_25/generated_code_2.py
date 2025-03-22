import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    inputs = layers.Input(shape=(32, 32, 3))

    # Path 1: Single 1x1 convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)

    # Path 2: Average pooling followed by a 1x1 convolution
    path2 = layers.AveragePooling2D(pool_size=(3, 3), strides=1)(inputs)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(path2)

    # Path 3: 1x1 convolution followed by 1x3 and 3x1 convolutions, then concatenation
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path3_1x3 = layers.Conv2D(32, (1, 3), activation='relu')(path3)
    path3_3x1 = layers.Conv2D(32, (3, 1), activation='relu')(path3)
    path3 = layers.Concatenate()([path3_1x3, path3_3x1])

    # Path 4: 1x1 convolution followed by 3x3 convolution, then 1x3 and 3x1 convolutions, then concatenation
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(inputs)
    path4 = layers.Conv2D(32, (3, 3), activation='relu')(path4)
    path4_1x3 = layers.Conv2D(32, (1, 3), activation='relu')(path4)
    path4_3x1 = layers.Conv2D(32, (3, 1), activation='relu')(path4)
    path4 = layers.Concatenate()([path4_1x3, path4_3x1])

    # Concatenate outputs of all paths
    merged = layers.Concatenate()([path1, path2, path3, path4])

    # Flatten the output and add fully connected layer for classification
    flatten = layers.Flatten()(merged)
    outputs = layers.Dense(10, activation='softmax')(flatten)

    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example usage
# model = dl_model()
# model.summary()