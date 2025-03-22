import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch1)
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(units=128, activation='relu')(branch1)
    branch1 = Dense(units=64, activation='relu')(branch1)
    branch1 = Dense(units=10, activation='softmax')(branch1)

    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(units=128, activation='relu')(branch2)
    branch2 = Dense(units=64, activation='relu')(branch2)
    branch2 = Dense(units=10, activation='softmax')(branch2)

    # Combine branches
    output = Concatenate()([branch1, branch2])
    output = Flatten()(output)
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model