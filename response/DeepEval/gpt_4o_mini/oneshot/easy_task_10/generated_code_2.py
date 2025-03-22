import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, Flatten, Dense

def dl_model():
    # Input layer for 28x28 grayscale images
    input_layer = Input(shape=(28, 28, 1))

    # 1x1 convolutional layer to increase dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(input_layer)
    
    # 3x3 depthwise separable convolutional layer for feature extraction
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding='valid', activation='relu')(conv1x1)

    # Another 1x1 convolutional layer to reduce dimensionality
    conv1x1_reduced = Conv2D(filters=16, kernel_size=(1, 1), strides=(2, 2), padding='valid', activation='relu')(depthwise_conv)

    # Flatten the output
    flatten_layer = Flatten()(conv1x1_reduced)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model