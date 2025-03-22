import keras
from keras.layers import Input, MaxPooling2D, Concatenate, Flatten, Dense, Conv2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Max pooling layers with varying scales
    max_pooling_1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    max_pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    max_pooling_3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    # Flatten and concatenate the outputs of the max pooling layers
    flatten_1 = Flatten()(max_pooling_1)
    flatten_2 = Flatten()(max_pooling_2)
    flatten_3 = Flatten()(max_pooling_3)
    concatenate = Concatenate()([flatten_1, flatten_2, flatten_3])

    # Block 2: Convolutional layers and max pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(concatenate)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
    max_pooling_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)

    # Flatten and concatenate the outputs of the convolutional layers and max pooling
    flatten_4 = Flatten()(max_pooling_4)
    concatenate_2 = Concatenate()([flatten_4, concatenate])

    # Output layer
    output_layer = Dense(10, activation='softmax')(concatenate_2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model