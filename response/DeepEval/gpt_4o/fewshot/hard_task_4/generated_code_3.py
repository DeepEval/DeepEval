import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Increase dimensionality of input channels threefold using 1x1 convolution
    expanded_channels = Conv2D(filters=9, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Step 2: Extract initial features using 3x3 depthwise separable convolution
    initial_features = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(expanded_channels)

    # Step 3: Compute channel attention weights
    global_pool = GlobalAveragePooling2D()(initial_features)
    fc1 = Dense(units=9 // 2, activation='relu')(global_pool)  # Reduce channel dimensions by half
    fc2 = Dense(units=9, activation='sigmoid')(fc1)  # Output channel attention weights
    channel_attention_weights = fc2

    # Step 4: Apply channel attention by reshaping and multiplying with initial features
    channel_attention_weights = keras.layers.Reshape((1, 1, 9))(channel_attention_weights)
    attention_applied = Multiply()([initial_features, channel_attention_weights])

    # Step 5: Reduce dimensionality with a 1x1 convolution
    reduced_dimensionality = Conv2D(filters=3, kernel_size=(1, 1), activation='relu', padding='same')(attention_applied)

    # Step 6: Combine with initial input
    combined_output = Add()([reduced_dimensionality, input_layer])

    # Step 7: Flatten and fully connect for final classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model