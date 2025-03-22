import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block
    block = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
    block = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block)

    # Branch 1
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
    branch1 = BatchNormalization()(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(units=128, activation='relu')(branch1)
    branch1 = Dense(units=64, activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
    branch2 = BatchNormalization()(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(units=128, activation='relu')(branch2)
    branch2 = Dense(units=64, activation='relu')(branch2)

    # Branch 3
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block)
    branch3 = BatchNormalization()(branch3)
    branch3 = Flatten()(branch3)
    branch3 = Dense(units=128, activation='relu')(branch3)
    branch3 = Dense(units=64, activation='relu')(branch3)

    # Output layer
    output = Concatenate()([branch1, branch2, branch3])
    output = Dense(units=10, activation='softmax')(output)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model