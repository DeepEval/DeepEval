from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first branch
    branch1 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = MaxPooling2D((2, 2))(branch1)

    # Define the second branch
    branch2 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D((2, 2))(branch2)

    # Define the third branch
    branch3 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch3 = Conv2D(64, (5, 5), activation='relu')(branch3)
    branch3 = MaxPooling2D((2, 2))(branch3)

    # Define the fourth branch
    branch4 = MaxPooling2D((2, 2))(input_shape)
    branch4 = Conv2D(64, (3, 3), activation='relu')(branch4)
    branch4 = MaxPooling2D((2, 2))(branch4)

    # Concatenate the branches
    x = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the features
    x = Flatten()(x)

    # Add a fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model