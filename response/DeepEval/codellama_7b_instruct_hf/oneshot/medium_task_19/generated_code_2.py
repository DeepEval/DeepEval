import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first branch
    branch_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_1)

    # Define the second branch
    branch_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_2)

    # Define the third branch
    branch_3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_3)

    # Define the fourth branch
    branch_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch_4)

    # Concatenate the branches
    concatenated = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Apply batch normalization and flatten the features
    normalized = BatchNormalization()(concatenated)
    flattened = Flatten()(normalized)

    # Add two fully connected layers to complete the classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model