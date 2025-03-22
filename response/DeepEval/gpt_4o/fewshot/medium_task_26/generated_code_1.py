import keras
from keras.layers import Input, Conv2D, Flatten, Concatenate, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Compress the input channels with a 1x1 convolution
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel convolutional layers
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

    # Concatenate the results
    concatenated = Concatenate()([conv1x1, conv3x3])

    # Flatten the feature map
    flatten = Flatten()(concatenated)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model