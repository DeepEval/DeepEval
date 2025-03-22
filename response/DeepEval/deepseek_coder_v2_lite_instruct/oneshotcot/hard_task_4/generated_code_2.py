import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))

    # Step 2: Add convolutional layer to increase channels by 3x with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Intermediate features
    intermediate_features = depthwise_conv

    # Step 4: Compute channel attention weights
    gap = GlobalAveragePooling2D()(intermediate_features)
    dense1 = Dense(units=intermediate_features.shape[3] // 16, activation='relu')(gap)
    dense2 = Dense(units=intermediate_features.shape[3], activation='sigmoid')(dense1)
    reshaped_weights = Dense(units=intermediate_features.shape[3], activation='sigmoid')(dense2)
    reshaped_weights = keras.backend.expand_dims(reshaped_weights, axis=-1)

    # Multiply the initial features with the weights
    weighted_features = Multiply()([intermediate_features, reshaped_weights])

    # Step 5: Reduce dimensionality with a 1x1 convolution
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(weighted_features)

    # Step 6: Combine with the initial input
    added = Add()([conv2, conv1])

    # Step 7: Flatten the result
    flatten_layer = Flatten()(added)

    # Step 8: Add fully connected layer
    dense3 = Dense(units=256, activation='relu')(flatten_layer)

    # Step 9: Add output layer
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model