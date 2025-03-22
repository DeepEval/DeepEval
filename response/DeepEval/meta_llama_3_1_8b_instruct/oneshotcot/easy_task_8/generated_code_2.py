import keras
from keras.layers import Input, DepthwiseConv2D, Conv2D, Dropout, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    # Depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    # Feature extraction using a 1x1 convolution
    conv = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(depthwise_conv)
    conv = Dropout(0.2)(conv)  # Add dropout to mitigate overfitting

    # Another convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv)
    conv2 = Dropout(0.2)(conv2)  # Add dropout to mitigate overfitting

    # Flatten the output
    flatten_layer = Flatten()(conv2)
    
    # Add a fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model