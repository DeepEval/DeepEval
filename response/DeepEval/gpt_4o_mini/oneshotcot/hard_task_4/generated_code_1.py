import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Step 1: Increase the dimensionality of the input channels threefold with a 1x1 convolution
    initial_features = Conv2D(filters=3*32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_separable_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(initial_features)

    # Step 3: Compute channel attention weights through global average pooling
    global_avg_pool = GlobalAveragePooling2D()(depthwise_separable_conv)

    # Step 4: Fully connected layers for channel attention
    dense1 = Dense(units=32, activation='relu')(global_avg_pool)
    dense2 = Dense(units=3*32, activation='sigmoid')(dense1)  # Output size same as initial features channels

    # Step 5: Reshape the weights to match the initial features
    reshaped_weights = keras.layers.Reshape((1, 1, 3*32))(dense2)

    # Step 6: Multiply the initial features with the weights for channel attention
    channel_attention_output = Multiply()([initial_features, reshaped_weights])

    # Step 7: A 1x1 convolution to reduce the dimensionality
    reduced_features = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(channel_attention_output)

    # Step 8: Combine the output with the initial input
    combined_output = Add()([initial_features, reduced_features])

    # Step 9: Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model