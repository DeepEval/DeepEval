import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Step 1: Increase the number of channels
    initial_features = Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Depthwise separable convolution
    extracted_features = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(initial_features)

    # Step 3: Channel attention mechanism
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(extracted_features)

    # Fully connected layers to generate weights
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=9, activation='sigmoid')(dense1)  # Output size matches the channels of initial_features

    # Reshape weights to match the dimensions of initial_features
    attention_weights = Reshape((1, 1, 9))(dense2)

    # Step 4: Apply channel attention
    attended_features = Multiply()([initial_features, attention_weights])

    # Step 5: Reduce dimensionality with 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(attended_features)

    # Step 6: Combine with initial input
    combined_features = keras.layers.Add()([input_layer, reduced_features])

    # Step 7: Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model