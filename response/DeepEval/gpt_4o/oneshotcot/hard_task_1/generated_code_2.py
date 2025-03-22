import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Multiply, AveragePooling2D, MaxPooling2D, Concatenate, Activation
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 color channels
    input_layer = Input(shape=input_shape)

    # Initial convolutional layer
    initial_conv = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Block 1: Channel Attention Mechanism
    def block1(input_tensor):
        # Path1: Global average pooling followed by two fully connected layers
        gap = GlobalAveragePooling2D()(input_tensor)
        dense1_gap = Dense(units=input_shape[-1] // 2, activation='relu')(gap)
        dense2_gap = Dense(units=input_shape[-1], activation='relu')(dense1_gap)

        # Path2: Global max pooling followed by two fully connected layers
        gmp = GlobalMaxPooling2D()(input_tensor)
        dense1_gmp = Dense(units=input_shape[-1] // 2, activation='relu')(gmp)
        dense2_gmp = Dense(units=input_shape[-1], activation='relu')(dense1_gmp)

        # Add paths together and apply activation
        channel_attention_weights = Activation('relu')(Add()([dense2_gap, dense2_gmp]))

        # Apply channel attention weights
        channel_refined = Multiply()([input_tensor, channel_attention_weights])

        return channel_refined

    block1_output = block1(initial_conv)

    # Block 2: Spatial Attention Mechanism
    def block2(input_tensor, channel_refined_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=1, padding='same')(channel_refined_tensor)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(channel_refined_tensor)

        # Concatenate and apply 1x1 convolution + sigmoid activation
        concatenated = Concatenate(axis=-1)([avg_pool, max_pool])
        spatial_attention = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(concatenated)

        # Apply spatial attention
        spatial_refined = Multiply()([channel_refined_tensor, spatial_attention])

        return spatial_refined

    block2_output = block2(initial_conv, block1_output)

    # Additional branch to align channels
    additional_conv = Conv2D(filters=input_shape[-1], kernel_size=(1, 1), padding='same', activation='relu')(block2_output)

    # Final addition and activation
    final_output = Activation('relu')(Add()([additional_conv, initial_conv]))

    # Fully connected layer for classification
    flatten_layer = keras.layers.Flatten()(final_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model