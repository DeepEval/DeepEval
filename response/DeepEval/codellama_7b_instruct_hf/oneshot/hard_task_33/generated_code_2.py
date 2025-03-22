from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch1 = input_layer
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(units=128, activation='relu')(branch1)
    branch1 = Dense(units=10, activation='softmax')(branch1)

    # Second branch
    branch2 = input_layer
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(units=128, activation='relu')(branch2)
    branch2 = Dense(units=10, activation='softmax')(branch2)

    # Third branch
    branch3 = input_layer
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch3)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Flatten()(branch3)
    branch3 = Dense(units=128, activation='relu')(branch3)
    branch3 = Dense(units=10, activation='softmax')(branch3)

    # Concatenate branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(concatenated)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model