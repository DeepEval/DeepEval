import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input's channels threefold with a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Compute channel attention weights through global average pooling
    global_average_pooling = GlobalAveragePooling2D()(conv2)

    # Generate weights whose size is same as the channels of the initial features
    dense1 = Dense(units=6, activation='relu')(global_average_pooling)

    # Reshape the weights to match the initial features and multiply with the initial features
    dense2 = Dense(units=6, activation='relu')(dense1)
    attention_weights = Concatenate()([dense1, dense2])

    # Reduce the dimensionality of the initial features
    conv3 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention_weights)

    # Combine the output with the initial input
    output = Concatenate()([conv3, input_layer])

    # Pass the output through a flattening layer and a fully connected layer to complete the classification process
    flatten_layer = Flatten()(output)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model