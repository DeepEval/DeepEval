from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    branch_1 = Conv2D(32, (1, 1), activation='relu')(input_shape)

    # Define the second branch
    branch_2 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch_2 = Conv2D(32, (3, 3), activation='relu')(branch_2)

    # Define the third branch
    branch_3 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch_3 = Conv2D(32, (5, 5), activation='relu')(branch_3)

    # Define the fourth branch
    branch_4 = MaxPooling2D((3, 3))(input_shape)
    branch_4 = Conv2D(32, (1, 1), activation='relu')(branch_4)

    # Concatenate the outputs of the branches
    x = Concatenate()([branch_1, branch_2, branch_3, branch_4])

    # Flatten the features
    x = Flatten()(x)

    # Add fully connected layers
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    return model