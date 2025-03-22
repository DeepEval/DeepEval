import tensorflow as tf
from tensorflow.keras import layers, models

def dl_model():
    input_layer = layers.Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Path 1: Single 1x1 Convolution
    path1 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Path 2: Average Pooling followed by 1x1 Convolution
    path2 = layers.AveragePooling2D(pool_size=(2, 2))(input_layer)
    path2 = layers.Conv2D(32, (1, 1), activation='relu')(path2)

    # Path 3: 1x1 Convolution followed by two parallel 1x3 and 3x1 convolutions
    path3 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    path3_1 = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(path3)
    path3_2 = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(path3)
    path3 = layers.concatenate([path3_1, path3_2])

    # Path 4: 1x1 Convolution followed by 3x3 Convolution, then two parallel 1x3 and 3x1 convolutions
    path4 = layers.Conv2D(32, (1, 1), activation='relu')(input_layer)
    path4 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(path4)
    path4_1 = layers.Conv2D(32, (1, 3), padding='same', activation='relu')(path4)
    path4_2 = layers.Conv2D(32, (3, 1), padding='same', activation='relu')(path4)
    path4 = layers.concatenate([path4_1, path4_2])

    # Concatenate the outputs of all paths
    concatenated = layers.concatenate([path1, path2, path3, path4])

    # Global Average Pooling layer
    x = layers.GlobalAveragePooling2D()(concatenated)

    # Fully connected layer for classification
    x = layers.Dense(128, activation='relu')(x)
    output_layer = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10

    # Create the model
    model = models.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model