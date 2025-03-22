import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    # First convolutional block
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Dropout(rate=0.2)(conv1)

    # Second convolutional block
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv2 = Dropout(rate=0.2)(conv2)

    # Third convolutional block
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Restore number of channels
    conv4 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Branch path
    branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch = Dropout(rate=0.2)(branch)

    # Combine main and branch path
    concat = Concatenate()([conv4, branch])

    # Flatten layer
    flatten = Flatten()(concat)

    # Fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)
    dense = Dense(units=10, activation='softmax')(dense)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model