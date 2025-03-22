from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Conv2D(64, (1, 1), activation='relu')(input_shape)
    main_path = Conv2D(64, (3, 3), activation='relu')(main_path)

    # Define the branch paths
    branch1 = Conv2D(64, (3, 3), activation='relu')(main_path)
    branch2 = MaxPooling2D((2, 2))(main_path)
    branch3 = MaxPooling2D((2, 2))(main_path)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D((2, 2))(branch3)

    # Concatenate the outputs from the branch paths
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Apply a 1x1 convolutional layer to the concatenated output
    concatenated = Conv2D(64, (1, 1), activation='relu')(concatenated)

    # Define the output shape
    output_shape = (32, 32, 64)

    # Define the final output layer
    output = Flatten()(concatenated)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    return model