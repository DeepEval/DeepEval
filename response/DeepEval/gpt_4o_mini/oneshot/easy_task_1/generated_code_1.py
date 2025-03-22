import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First convolutional layer followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool1)

    # Flatten the output from the last convolutional layer
    flatten_layer = Flatten()(conv2)

    # First fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Second fully connected layer
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Output layer for classification
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model