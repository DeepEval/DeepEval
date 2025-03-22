import keras
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, Multiply, Flatten, Add, SeparableConv2D, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase dimensionality threefold using a 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Initial features with a 3x3 depthwise separable convolution
    initial_features = SeparableConv2D(filters=9, kernel_size=(3, 3), padding='same', activation='relu')(expanded_channels)

    # Global Average Pooling for channel attention
    gap = GlobalAveragePooling2D()(initial_features)
    
    # Fully connected layers to compute channel attention weights
    dense1 = Dense(units=9 // 2, activation='relu')(gap)
    dense2 = Dense(units=9, activation='sigmoid')(dense1)

    # Reshape and apply channel attention
    attention_weights = Reshape((1, 1, 9))(dense2)
    channel_attention = Multiply()([initial_features, attention_weights])

    # Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(channel_attention)

    # Combine with the initial input
    combined = Add()([reduced_features, input_layer])

    # Flatten and final fully connected layer for classification
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model