import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Flatten, Dense, Add, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # 1x1 convolution to increase channel dimensionality
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolution for extracting initial features
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Global average pooling to compute channel attention weights
    pooling = GlobalAveragePooling2D()(conv2)

    # Two fully connected layers for generating channel attention weights
    dense1 = Dense(units=8, activation='relu')(pooling)
    dense2 = Dense(units=3, activation='relu')(dense1)

    # Reshaping the weights to match the initial features
    reshaped_weights = Reshape(target_shape=(3, 1, 1))(dense2)

    # Multiplying the weights with the initial features to achieve channel attention weighting
    attention_weights = Add()([conv2, reshaped_weights])

    # 1x1 convolution to reduce channel dimensionality
    conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_weights)

    # Combining the output of the convolution with the initial input
    combined_output = Add()([conv3, input_layer])

    # Flattening the output and passing through a fully connected layer for classification
    flattened_output = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flattened_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model