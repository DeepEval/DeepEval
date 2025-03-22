import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, DepthwiseConv2D

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    # Dropout to prevent overfitting
    conv1 = Dropout(0.2)(conv1)

    # 1x1 convolutional layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Dropout(0.2)(conv2)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(conv2)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model