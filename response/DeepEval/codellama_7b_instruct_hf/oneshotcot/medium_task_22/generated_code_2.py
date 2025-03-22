import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch with 3x3 convolutions
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)

    # Define the second branch with 1x1 convolutions followed by two 3x3 convolutions
    branch_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_shape)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)

    # Define the third branch with max pooling
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_shape)

    # Concatenate the outputs from the three branches
    concatenated = Concatenate()([branch_1, branch_2, branch_3])

    # Apply batch normalization and flatten the result
    normalized = BatchNormalization()(concatenated)
    flattened = Flatten()(normalized)

    # Add two fully connected layers for classification
    fc_1 = Dense(units=128, activation='relu')(flattened)
    fc_2 = Dense(units=64, activation='relu')(fc_1)
    output = Dense(units=10, activation='softmax')(fc_2)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output)

    return model