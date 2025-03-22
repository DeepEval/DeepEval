import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 channels

    # Step 2: 1x1 convolution to increase channel dimensionality
    conv1x1 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 3: 3x3 depthwise separable convolution to extract features
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1)

    # Step 4: Global Average Pooling to compute channel attention weights
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)

    # Step 5: Two fully connected layers to generate weights
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=96, activation='sigmoid')(dense1)  # Size matches the number of channels in conv1x1

    # Step 6: Reshape weights to match the initial feature map dimensions
    channel_weights = Reshape((1, 1, 96))(dense2)

    # Step 7: Multiply weights with the initial features for channel attention weighting
    attention_output = Multiply()([depthwise_conv, channel_weights])

    # Step 8: 1x1 convolution to reduce dimensionality
    conv_reduce = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_output)

    # Step 9: Combine the output with the initial input (residual connection)
    combined_output = Add()([input_layer, conv_reduce])

    # Step 10: Flatten layer followed by a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # 10 classes for CIFAR-10

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model