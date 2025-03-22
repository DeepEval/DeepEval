import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Split

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    concatenated = Concatenate()([flatten1, flatten2, flatten3])

    # Fully connected layer and reshape operation
    dense = Dense(units=128, activation='relu')(concatenated)
    reshaped = Reshape((128, 1))(dense)

    # Second block
    split = Split(num_or_size_splits=4)(reshaped)
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
    conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split[3])
    concatenated_outputs = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten and fully connected layer
    flattened = Flatten()(concatenated_outputs)
    dense2 = Dense(units=64, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model