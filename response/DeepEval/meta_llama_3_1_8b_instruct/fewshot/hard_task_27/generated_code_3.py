import keras
from keras.layers import Input, DepthwiseConv2D, BatchNormalization, Lambda, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Extract spatial features using a 7x7 depthwise separable convolutional layer
    conv1 = DepthwiseConv2D(kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Apply layer normalization to enhance training stability
    bn = BatchNormalization()(conv1)

    # Pass the output through two fully connected layers with the same numbers of channel as the input layer
    conv2 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(bn)
    conv3 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Combine the original input with the processed features through an addition operation
    adding_layer = Lambda(lambda x: x + input_layer)(conv3)

    # Pass the output through two fully connected layers for classification
    flatten = Flatten()(adding_layer)
    dense1 = Dense(units=384, activation='relu')(flatten)
    dense2 = Dense(units=192, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model