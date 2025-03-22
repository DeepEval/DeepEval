from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, concatenate, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_img = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (1, 1), padding='same')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = Conv2D(64, (1, 1), padding='same')(pool1)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    pool2 = MaxPooling2D(pool_size=(2, 2))(x)

    # Block 3
    x = Conv2D(128, (1, 1), padding='same')(pool2)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, (5, 5), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    pool3 = MaxPooling2D(pool_size=(2, 2))(x)

    # Concatenation
    concat = concatenate([pool1, pool2, pool3])

    # Fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(512, activation='relu')(flatten)
    output = Dense(10, activation='softmax')(dense1)

    # Model definition
    model = Model(input_img, output)

    return model