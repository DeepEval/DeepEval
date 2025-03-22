import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Conv2D, Add, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    # 1x1 convolutional layer to increase dimensionality
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # 3x3 depthwise separable convolutional layer for feature extraction
    conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # 1x1 convolutional layer to reduce dimensionality
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Add the output from the last layer to the original input layer
    output = Add()([input_layer, conv3])

    # Batch normalization
    batch_norm = BatchNormalization()(output)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layer to generate the final classification probabilities
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model