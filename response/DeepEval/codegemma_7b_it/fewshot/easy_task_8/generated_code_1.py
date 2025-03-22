import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Lambda, Flatten, Concatenate, DepthwiseConv2D, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    conv1 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Feature extraction with 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Dropout to mitigate overfitting
    conv2 = Dropout(0.3)(conv2)

    # Max pooling for feature extraction
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(max_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    # Build and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model