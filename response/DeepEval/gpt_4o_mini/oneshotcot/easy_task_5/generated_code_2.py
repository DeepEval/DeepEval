import keras
from keras.layers import Input, Conv2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST images are 28x28 pixels with 1 channel

    # Step 2: Add 1x1 convolutional layer to reduce dimensionality
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Step 3: Add 3x3 convolutional layer to extract features
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)

    # Step 4: Add another 1x1 convolutional layer to restore dimensionality
    conv1x1_restore = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(conv3x3)

    # Step 5: Flatten the output
    flatten_layer = Flatten()(conv1x1_restore)

    # Step 6: Add fully connected layer with 10 neurons for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model