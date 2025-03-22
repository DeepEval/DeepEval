import keras
from keras.layers import Input, DepthwiseSeparableConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense


def dl_model():
    # Import necessary packages
    import keras
    from keras.layers import Input, DepthwiseSeparableConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # Add depthwise separable convolutional layer
    conv1 = DepthwiseSeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid')(input_layer)

    # Add 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(conv1)

    # Add dropout layer
    dropout = Dropout(rate=0.2)(conv2)

    # Add max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(dropout)

    # Flatten output
    flatten = Flatten()(max_pooling)

    # Add fully connected layer
    fc = Dense(units=128, activation='relu')(flatten)

    # Add dropout layer
    dropout = Dropout(rate=0.5)(fc)

    # Add fully connected layer
    fc = Dense(units=10, activation='softmax')(dropout)

    # Define model
    model = keras.Model(inputs=input_layer, outputs=fc)

    # Return model
    return model