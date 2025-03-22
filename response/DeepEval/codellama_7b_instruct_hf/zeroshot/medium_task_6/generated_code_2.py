from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)

    # Parallel blocks
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Addition of parallel blocks' outputs and initial convolution's output
    x = Add()([x, x])

    # Flattening
    x = Flatten()(x)

    # Fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create model
    model = Model(inputs=input_layer, outputs=x)

    return model