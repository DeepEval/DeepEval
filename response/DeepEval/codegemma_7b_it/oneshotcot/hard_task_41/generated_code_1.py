import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input_layer)

    path1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(path1)
    path2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(path2)
    path3 = MaxPooling2D(pool_size=(4, 4), strides=1, padding='same')(path3)

    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)

    path1 = Dropout(rate=0.25)(path1)
    path2 = Dropout(rate=0.25)(path2)
    path3 = Dropout(rate=0.25)(path3)

    concat_block1 = Concatenate()([path1, path2, path3])

    # Block 2
    reshape_layer = Reshape((28, 28, 1))(concat_block1)

    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(reshape_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    branch1 = Flatten()(branch1)
    branch2 = Flatten()(branch2)
    branch3 = Flatten()(branch3)
    branch4 = Flatten()(branch4)

    concat_block2 = Concatenate()([branch1, branch2, branch3, branch4])

    # Output layer
    dense1 = Dense(units=128, activation='relu')(concat_block2)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model definition
    model = Model(inputs=input_layer, outputs=output_layer)

    return model