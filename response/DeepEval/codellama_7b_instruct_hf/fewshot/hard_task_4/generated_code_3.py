import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Flatten, Dense, Lambda, Concatenate
from keras.applications.vgg16 import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Compute channel attention weights through global average pooling
    pool = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=16, activation='relu')(pool)
    dense2 = Dense(units=8, activation='relu')(dense1)

    # Generate weights whose size is same as the channels of the initial features
    channel_attention_weights = Dense(units=64, activation='softmax')(dense2)

    # Reshape the channel attention weights to match the initial features
    reshaped_channel_attention_weights = Lambda(lambda x: tf.reshape(x, shape=(64, 64)))(channel_attention_weights)

    # Multiply the channel attention weights with the initial features to achieve channel attention weighting
    attention_output = Concatenate()([conv2, reshaped_channel_attention_weights])

    # Apply a 1x1 convolution to reduce the dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(attention_output)

    # Combine the output with the initial input
    combined_output = Concatenate()([conv3, input_layer])

    # Pass through a flattening layer and a fully connected layer to complete the classification process
    flatten = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model