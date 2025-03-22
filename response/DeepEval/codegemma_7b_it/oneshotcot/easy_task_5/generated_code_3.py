import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Reduce dimensionality with 1x1 convolution
    conv_1x1 = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract features using 3x3 convolutional layer
    conv_3x3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_1x1)

    # Restore dimensionality with another 1x1 convolution layer
    conv_1x1_restore = Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_3x3)

    # Flatten the output
    flatten_layer = Flatten()(conv_1x1_restore)

    # Fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model