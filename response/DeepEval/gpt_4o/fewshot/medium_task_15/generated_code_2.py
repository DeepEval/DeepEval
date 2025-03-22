import keras
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Dense, Multiply, Concatenate, AveragePooling2D, Flatten

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer with batch normalization and ReLU activation
    initial_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    initial_bn = BatchNormalization()(initial_conv)
    initial_relu = Activation('relu')(initial_bn)

    # Global average pooling to compress feature maps
    global_avg_pooling = GlobalAveragePooling2D()(initial_relu)

    # Fully connected layers to adjust dimensions
    fc1 = Dense(units=64, activation='relu')(global_avg_pooling)
    fc2 = Dense(units=64, activation='sigmoid')(fc1)

    # Reshape to match the initial feature size
    reshaped = Reshape(target_shape=(1, 1, 64))(fc2)

    # Multiply with initial features to generate weighted feature maps
    weighted_features = Multiply()([initial_relu, reshaped])

    # Concatenate weighted features with input layer
    concatenated = Concatenate(axis=-1)([input_layer, weighted_features])

    # Reduce dimensionality and downsample using 1x1 convolution and average pooling
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    downsampled = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(compressed)

    # Final fully connected layer for classification
    flatten = Flatten()(downsampled)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Create the Keras model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model