import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Reshape, Flatten, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels

    # Step 1: Increase dimensionality of the input's channels threefold with a 1x1 convolution
    initial_features = Conv2D(filters=3 * 32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(initial_features)

    # Step 3: Compute channel attention weights through global average pooling
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3 * 32, activation='sigmoid')(dense1)  # Output same size as channels of initial features

    # Step 4: Reshape weights to match initial features and apply channel attention
    attention_weights = Reshape((1, 1, 3 * 32))(dense2)  # Reshape to match dimensions
    channel_attention = Multiply()([initial_features, attention_weights])  # Apply attention

    # Step 5: Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channel_attention)

    # Step 6: Combine the output with the initial input using an Add layer
    combined_output = Add()([initial_features, reduced_features])

    # Step 7: Flatten the output and add a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model