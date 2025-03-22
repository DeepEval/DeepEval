import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Reshape, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Initial convolutional layer
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Two fully connected layers
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)

    # Adjust dimensions to match the channels of the initial features
    adjusted_features = Reshape((1, 1, 64))(x)

    # Generate weighted feature maps by multiplying with the initial features
    weighted_features = Multiply()([input_layer, adjusted_features])

    # Concatenate weighted feature maps with the input layer
    concatenated_features = Concatenate()([input_layer, weighted_features])

    # 1x1 Convolution and Average Pooling to reduce dimensionality and downsample the feature maps
    x = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(concatenated_features)
    x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # Flatten and output layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model