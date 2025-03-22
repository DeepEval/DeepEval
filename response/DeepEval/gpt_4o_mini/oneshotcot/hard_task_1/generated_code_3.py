import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, BatchNormalization, Concatenate
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 input shape

    # Initial convolutional layer to adjust output channels
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Block 1
    # Path 1: Global average pooling followed by two fully connected layers
    gap1 = GlobalAveragePooling2D()(conv1)
    dense1a = Dense(units=32, activation='relu')(gap1)
    dense1b = Dense(units=32, activation='relu')(dense1a)

    # Path 2: Global max pooling followed by two fully connected layers
    gmp1 = GlobalMaxPooling2D()(conv1)
    dense2a = Dense(units=32, activation='relu')(gmp1)
    dense2b = Dense(units=32, activation='relu')(dense2a)

    # Channel attention weights
    channel_attention = Add()([dense1b, dense2b])
    channel_attention = Activation('sigmoid')(channel_attention)
    channel_attention = Dense(units=32, activation='sigmoid')(channel_attention)  # Adjusting to match input channels

    # Element-wise multiplication with original features
    attention_output = Multiply()([conv1, channel_attention])

    # Block 2: Spatial feature extraction
    avg_pool = GlobalAveragePooling2D()(attention_output)
    max_pool = GlobalMaxPooling2D()(attention_output)

    # Concatenate the outputs along the channel dimension
    concatenated = Concatenate()([avg_pool, max_pool])
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='sigmoid')(concatenated)

    # Element-wise multiplication with channel features
    spatial_output = Multiply()([attention_output, conv2])

    # Final branch: 1x1 convolution to align output channels with input channels
    final_conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(spatial_output)

    # Add to main path and activate
    final_output = Add()([attention_output, final_conv])
    final_output = Activation('relu')(final_output)

    # Fully connected layer for classification
    flatten_layer = GlobalAveragePooling2D()(final_output)  # Flatten the output
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model