import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():  
    input_layer = Input(shape=(28, 28, 1))

    # Reduce dimensionality with 1x1 convolution
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Extract features with 3x3 convolution
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Restore dimensionality with 1x1 convolution
    conv3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Flatten the output
    flatten = Flatten()(conv3)

    # Output layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model