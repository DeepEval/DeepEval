import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution, 3x3 convolution
    branch_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
    branch_1 = Dropout(0.2)(branch_1)

    # Branch 2: 1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution
    branch_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Dropout(0.2)(branch_2)

    # Branch 3: max pooling
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)
    branch_3 = Dropout(0.2)(branch_3)

    # Concatenate branches
    output = Concatenate()([branch_1, branch_2, branch_3])

    # Batch normalization
    output = BatchNormalization()(output)

    # Flatten and fully connected layers
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(64, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create and return model
    model = Model(inputs=input_layer, outputs=output)
    return model