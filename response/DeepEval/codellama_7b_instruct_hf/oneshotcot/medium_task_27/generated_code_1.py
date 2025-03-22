import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First parallel branch
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(units=128, activation='relu')(branch1)
    branch1 = Dense(units=64, activation='relu')(branch1)
    branch1 = Dense(units=10, activation='softmax')(branch1)

    # Second parallel branch
    branch2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(units=128, activation='relu')(branch2)
    branch2 = Dense(units=64, activation='relu')(branch2)
    branch2 = Dense(units=10, activation='softmax')(branch2)

    # Combine branches
    branch_output = Concatenate()([branch1, branch2])

    # Global average pooling
    branch_output = GlobalAveragePooling2D()(branch_output)

    # Fully connected layers
    branch_output = Dense(units=128, activation='relu')(branch_output)
    branch_output = Dense(units=64, activation='relu')(branch_output)
    branch_output = Dense(units=10, activation='softmax')(branch_output)

    model = keras.Model(inputs=input_layer, outputs=branch_output)

    return model