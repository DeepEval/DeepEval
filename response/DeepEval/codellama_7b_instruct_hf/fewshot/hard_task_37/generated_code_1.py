import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first branch
    branch_1 = input_layer
    for i in range(3):
        branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
        branch_1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch_1)

    # Define the second branch
    branch_2 = input_layer
    for i in range(3):
        branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
        branch_2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch_2)

    # Combine the branches
    branches = Add()([branch_1, branch_2])

    # Flatten the output and add a fully connected layer
    flatten = Flatten()(branches)
    output = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model