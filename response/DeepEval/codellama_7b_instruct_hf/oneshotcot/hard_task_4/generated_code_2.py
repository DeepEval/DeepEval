import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, BatchNormalization
from keras.models import Model

def dl_model():
    # Define the input layer with 3 channels
    input_layer = Input(shape=(32, 32, 3))

    # Increase the dimensionality of the input channels by a 1x1 convolution
    conv1 = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract initial features using a 3x3 depthwise separable convolution
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Compute channel attention weights through global average pooling
    pooling = GlobalAveragePooling2D()(conv2)

    # Generate weights whose size is same as the channels of the initial features
    dense1 = Dense(units=conv2.shape[1], activation='relu')(pooling)

    # Reshape the weights to match the initial features
    dense2 = Dense(units=conv2.shape[1], activation='relu')(dense1)

    # Multiply the weights with the initial features to achieve channel attention weighting
    attention = Dense(units=conv2.shape[1], activation='relu')(dense2)

    # Reduce the dimensionality of the output by a 1x1 convolution
    conv3 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(attention)

    # Combine the initial input with the output of the 1x1 convolution
    output = keras.layers.concatenate([input_layer, conv3])

    # Flatten the output and pass it through a fully connected layer to complete the classification process
    flatten = Flatten()(output)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Define the model
    model = Model(inputs=input_layer, outputs=dense)

    return model