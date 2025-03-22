import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 1: 1x1 convolution to increase the number of channels
    conv1x1 = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Depthwise separable convolution to extract features
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(conv1x1)

    # Step 3: Global Average Pooling for channel attention
    gap = GlobalAveragePooling2D()(depthwise_conv)

    # Step 4: Two fully connected layers to compute channel attention weights
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=9, activation='sigmoid')(dense1)  # Output size matches the number of channels in conv1x1

    # Step 5: Reshape weights to match the dimensions of the initial features
    attention_weights = Reshape((1, 1, 9))(dense2)

    # Step 6: Apply channel attention weighting
    weighted_features = Multiply()([conv1x1, attention_weights])

    # Step 7: 1x1 convolution to reduce dimensionality
    conv1x1_final = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(weighted_features)

    # Step 8: Combine the output with the initial input
    combined_output = Add()([input_layer, conv1x1_final])

    # Step 9: Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    model = Model(inputs=input_layer, outputs=output_layer)

    return model