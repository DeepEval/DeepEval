import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 64))

    # Step 1: Compressing the input channels with a 1x1 convolutional layer
    compressed = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Step 2: Expanding features through two parallel convolutional layers
    conv_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(compressed)
    conv_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(compressed)

    # Step 3: Concatenating the results of the two parallel layers
    concatenated = Concatenate()([conv_1x1, conv_3x3])

    # Step 4: Flattening the output feature map into a one-dimensional vector
    flattened = Flatten()(concatenated)

    # Step 5: Passing through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for classification

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model