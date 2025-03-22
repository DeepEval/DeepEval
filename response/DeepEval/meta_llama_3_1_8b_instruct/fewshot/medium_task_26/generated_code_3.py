import keras
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 64))

    # Compress the input channels with a 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Expand the features through two parallel convolutional layers
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Concatenate the results
    output_tensor = Concatenate()([conv2, conv3])

    # Flatten the output feature map
    flatten_layer = Flatten()(output_tensor)

    # Pass through two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model