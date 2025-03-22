import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # Compress the input channels with a 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Expand the features through two parallel convolutional layers
    conv2a = Conv2D(32, (1, 1), padding='same', activation='relu')(conv1)
    conv2b = Conv2D(32, (3, 3), padding='same', activation='relu')(conv1)
    concatenated = Concatenate()([conv2a, conv2b])

    # Flatten the output feature map
    flattened = Flatten()(concatenated)

    # Pass the flattened output through two fully connected layers
    dense1 = Dense(64, activation='relu')(flattened)
    output_layer = Dense(10, activation='softmax')(dense1)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model