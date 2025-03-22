import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Add the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Add the max-pooling layer
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Add the direct connection between input and max-pooling output
    concat = Concatenate()([input_layer, max_pool])

    # Add batch normalization
    batch_norm = BatchNormalization()(concat)

    # Flatten the output
    flatten = Flatten()(batch_norm)

    # Add the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Add the second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Add the output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model