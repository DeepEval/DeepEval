import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layer with batch normalization and ReLU activation
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Compress feature maps using global average pooling
    x = GlobalAveragePooling2D()(x)

    # Two fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    # Reshape to match the channels of the initial features
    x = Reshape((1, 1, 64))(x)

    # Multiply with the initial features to generate weighted feature maps
    weighted_feature_maps = Multiply()([input_layer, x])

    # Concatenate weighted feature maps with the input layer
    concatenated = Concatenate()([input_layer, weighted_feature_maps])

    # Reduce dimensionality and downsample the feature using 1x1 convolution and average pooling
    x = Conv2D(32, (1, 1), activation='relu')(concatenated)
    x = AveragePooling2D((2, 2))(x)

    # Final fully connected layer
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage:
# model = dl_model()
# model.summary()