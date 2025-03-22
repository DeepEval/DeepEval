from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, concatenate,
                         Activation, BatchNormalization, Lambda)

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Feature extraction path 1: 1x1 convolution
    tower_1x1 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)

    # Feature extraction path 2: 1x7, 7x1 convolution
    tower_1x7 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(inputs)
    tower_1x7 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(tower_1x7)

    # Feature extraction path 3: 1x7, 7x1 convolution
    tower_7x1 = Conv2D(64, (7, 1), padding='same', kernel_initializer='he_normal')(inputs)
    tower_7x1 = Conv2D(64, (1, 7), padding='same', kernel_initializer='he_normal')(tower_7x1)

    # Concatenate feature extraction paths
    concat_path = concatenate([tower_1x1, tower_1x7, tower_7x1], axis=-1)

    # Main path: 1x1 convolution to match input channel
    conv_path = Conv2D(3, (1, 1), padding='same', kernel_initializer='he_normal')(concat_path)

    # Branch connecting directly to input
    branch = Lambda(lambda x: x)(inputs)

    # Merge outputs of main path and branch
    merged = add([conv_path, branch])

    # Fully connected layers for classification
    x = Activation('relu')(merged)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(64, kernel_initializer='he_normal')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(10, activation='softmax', kernel_initializer='he_normal')(x)

    model = Model(inputs, predictions)

    return model