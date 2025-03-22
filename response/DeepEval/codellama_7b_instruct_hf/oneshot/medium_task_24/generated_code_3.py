import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Dropout, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first branch
    branch_1 = Conv2D(32, (1, 1), padding='same')(input_layer)
    branch_1 = Conv2D(64, (3, 3), padding='same')(branch_1)
    branch_1 = Dropout(0.2)(branch_1)

    # Define the second branch
    branch_2 = Conv2D(32, (1, 1), padding='same')(input_layer)
    branch_2 = Conv2D(64, (1, 7), padding='same')(branch_2)
    branch_2 = Conv2D(64, (7, 1), padding='same')(branch_2)
    branch_2 = Conv2D(64, (3, 3), padding='same')(branch_2)
    branch_2 = Dropout(0.2)(branch_2)

    # Define the third branch
    branch_3 = MaxPooling2D((2, 2), padding='same')(input_layer)
    branch_3 = Dropout(0.2)(branch_3)

    # Concatenate the outputs from all branches
    merged = Concatenate()([branch_1, branch_2, branch_3])

    # Flatten the merged output
    flattened = Flatten()(merged)

    # Add three fully connected layers
    dense_1 = Dense(64, activation='relu')(flattened)
    dense_2 = Dense(32, activation='relu')(dense_1)
    output_layer = Dense(10, activation='softmax')(dense_2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model