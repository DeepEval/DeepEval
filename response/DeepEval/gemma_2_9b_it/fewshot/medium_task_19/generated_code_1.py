import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)

    # Branch 2
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)

    # Branch 3
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(branch3)

    # Branch 4
    branch4 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(input_layer)
    branch4 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(branch4)

    # Concatenate branches
    merged = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    flatten = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model