import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Define the first branch
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch1)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch1)
    branch1 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch1)

    # Define the second branch
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch2)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch2)
    branch2 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(branch2)

    # Combine the outputs from both branches
    merged = Add()([branch1, branch2])

    # Flatten the output and pass through a fully connected layer
    flattened = Flatten()(merged)
    dense = Dense(units=128, activation='relu')(flattened)
    output = Dense(units=10, activation='softmax')(dense)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model