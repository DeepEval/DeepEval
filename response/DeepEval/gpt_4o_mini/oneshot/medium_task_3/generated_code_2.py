import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Convolution + MaxPooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second block: Convolution + MaxPooling
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Ensure that the output channel dimension is the same as the input dimension
    # In this case, it's (28, 28, 1) so we need to adjust the channels accordingly
    # Since the first pooling reduces dimensions, we need to upsample it or adjust the output
    # Here we'll use a Conv2D layer to match the dimensions for addition
    conv_for_addition = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)

    # Add the output of the second block to the adjusted input
    added_output = Add()([pool2, conv_for_addition])

    # Flatten the output and pass it through a Dense layer for classification
    flatten_layer = Flatten()(added_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model