import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # Adjusted shape to accommodate color images
    channel_split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Feature extraction for each channel
    channel1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_split[0])
    channel2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_split[1])
    channel3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_split[2])

    # Concatenate the outputs of the three groups
    concatenated = Concatenate()(list(channel1, channel2, channel3))

    # Flatten and pass through three fully connected layers
    flattened = Flatten()(concatenated)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model