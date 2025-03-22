import keras
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))

    # Step 2: Add the first 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    drop1 = Dropout(0.25)(conv1)

    # Step 3: Add the second 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop1)
    drop2 = Dropout(0.25)(conv2)

    # Step 4: Add the 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(drop2)
    drop3 = Dropout(0.25)(conv3)

    # Step 5: Add the 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(drop3)
    drop4 = Dropout(0.25)(conv4)

    # Step 6: Restore the number of channels with another 1x1 convolutional layer
    conv5 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(drop4)

    # Step 7: Combine features with the original input via addition
    added_output = Add()([input_layer, conv5])

    # Step 8: Add flatten layer
    flatten_layer = Flatten()(added_output)

    # Step 9: Add fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model