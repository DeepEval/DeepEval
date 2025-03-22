import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 pixels with 3 color channels.

    # Step 1: Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    initial_features = Conv2D(filters=9, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Extract initial features using a 3x3 depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(initial_features)

    # Step 3: Compute channel attention weights through global average pooling
    global_avg_pool = GlobalAveragePooling2D()(depthwise_conv)
    dense1 = Dense(units=9, activation='relu')(global_avg_pool)  # Fully connected layer to learn weights
    dense2 = Dense(units=9, activation='sigmoid')(dense1)  # Sigmoid activation for weights (0 to 1)

    # Step 4: Reshape the weights to match the initial features channels and multiply
    reshaped_weights = Reshape((1, 1, 9))(dense2)
    channel_attention = Multiply()([initial_features, reshaped_weights])  # Apply channel attention

    # Step 5: Reduce dimensionality with a 1x1 convolution
    reduced_features = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(channel_attention)

    # Step 6: Combine the output with the initial input
    combined_output = Add()([input_layer, reduced_features])  # Skip connection

    # Step 7: Flatten and pass through a fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)  # CIFAR-10 has 10 classes

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model