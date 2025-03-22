import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # First sequential block: three convolutional layers followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Second sequential block: four convolutional layers followed by max pooling
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pooling)
    conv5 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv4)
    conv6 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv5)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)

    # Flatten the output of the second sequential block
    flatten_layer = Flatten()(max_pooling2)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model