from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Path 1: 1x1 convolution
    path_1 = layers.Conv2D(32, (1, 1), padding='same')(input_layer)

    # Path 2: Average pooling + 1x1 convolution
    path_2 = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    path_2 = layers.Conv2D(32, (1, 1), padding='same')(path_2)

    # Path 3: 1x1 convolution + 1x3 and 3x1 convolutions
    path_3 = layers.Conv2D(32, (1, 1), padding='same')(input_layer)
    path_3_1x3 = layers.Conv2D(32, (1, 3), padding='same')(path_3)
    path_3_3x1 = layers.Conv2D(32, (3, 1), padding='same')(path_3)
    path_3 = layers.Concatenate()([path_3_1x3, path_3_3x1])

    # Path 4: 1x1 convolution + 3x3 convolution + 1x3 and 3x1 convolutions
    path_4 = layers.Conv2D(32, (1, 1), padding='same')(input_layer)
    path_4 = layers.Conv2D(32, (3, 3), padding='same')(path_4)
    path_4_1x3 = layers.Conv2D(32, (1, 3), padding='same')(path_4)
    path_4_3x1 = layers.Conv2D(32, (3, 1), padding='same')(path_4)
    path_4 = layers.Concatenate()([path_4_1x3, path_4_3x1])

    # Multi-scale feature fusion
    concat_features = layers.Concatenate()([path_1, path_2, path_3, path_4])

    # Fully connected layer for classification
    fc_layer = layers.Dense(10, activation='softmax')(concat_features)

    # Model construction
    model = models.Model(inputs=input_layer, outputs=fc_layer)

    return model