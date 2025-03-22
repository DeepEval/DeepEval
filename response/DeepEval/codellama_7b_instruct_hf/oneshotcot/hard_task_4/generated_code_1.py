import keras
from keras.layers import Input, Conv2D, DepthwiseSeparableConv2D, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Increase dimensionality with 1x1 convolution
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features with 3x3 depthwise separable convolution
    conv2 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Compute channel attention weights with global average pooling
    gavg_pooling = GlobalAveragePooling2D()(conv2)
    attn_weights = Dense(units=32, activation='softmax')(gavg_pooling)

    # Reshape attention weights to match initial features
    attn_weights_reshaped = Flatten()(attn_weights)

    # Multiply attention weights with initial features to achieve channel attention weighting
    attn_output = tf.math.multiply(conv2, attn_weights_reshaped)

    # Reduce dimensionality with 1x1 convolution
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attn_output)

    # Combine output with initial input
    output = tf.math.add(conv3, input_layer)

    # Flatten output and pass through fully connected layer
    flattened_output = Flatten()(output)
    dense_output = Dense(units=10, activation='softmax')(flattened_output)

    # Define and return model
    model = keras.Model(inputs=input_layer, outputs=dense_output)
    return model