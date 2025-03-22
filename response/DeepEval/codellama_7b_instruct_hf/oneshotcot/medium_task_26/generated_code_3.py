from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))

    # Convolutional layer
    conv1 = Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2)(conv1)

    # Parallel convolutional layers
    conv2_1 = Conv2D(128, (1, 1), activation='relu')(max_pooling)
    conv2_2 = Conv2D(128, (3, 3), activation='relu')(max_pooling)

    # Concatenate
    concatenated = Concatenate()([conv2_1, conv2_2])

    # Flatten
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flattened)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model