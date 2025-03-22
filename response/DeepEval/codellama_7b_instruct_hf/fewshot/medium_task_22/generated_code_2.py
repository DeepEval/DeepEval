import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first branch with 3x3 convolutions
    branch_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
    branch_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_1)

    # Define the second branch with 1x1 convolutions and two 3x3 convolutions
    branch_2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_2)

    # Define the third branch with max pooling
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch_1, branch_2, branch_3])

    # Flatten the concatenated outputs and pass through two fully connected layers for classification
    flattened = Flatten()(concatenated)
    output_layer = Dense(units=10, activation='softmax')(flattened)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model