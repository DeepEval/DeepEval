import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Softmax
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch of the model
    branch_1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_1 = Conv2D(64, (5, 5), activation='relu')(branch_1)
    branch_1 = GlobalAveragePooling2D()(branch_1)
    branch_1 = Dense(64, activation='relu')(branch_1)
    branch_1 = Dense(10, activation='softmax')(branch_1)

    # Define the second branch of the model
    branch_2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_2 = Conv2D(64, (5, 5), activation='relu')(branch_2)
    branch_2 = GlobalAveragePooling2D()(branch_2)
    branch_2 = Dense(64, activation='relu')(branch_2)
    branch_2 = Dense(10, activation='softmax')(branch_2)

    # Combine the two branches
    merged_outputs = Softmax()(branch_1, branch_2)

    # Define the model
    model = Model(inputs=input_shape, outputs=merged_outputs)

    return model