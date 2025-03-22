from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, concatenate
from keras.applications import VGG16

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the branches for the input
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    branch1 = Conv2D(64, (3, 3), activation='relu')(branch1)
    branch1 = Dropout(0.2)(branch1)

    branch2 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(64, (1, 7), activation='relu')(branch2)
    branch2 = Conv2D(64, (7, 1), activation='relu')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    branch3 = MaxPooling2D((2, 2))(input_shape)
    branch3 = Flatten()(branch3)

    # Concatenate the outputs from all branches
    x = concatenate([branch1, branch2, branch3])

    # Add a dense layer with a ReLU activation function
    x = Dense(128, activation='relu')(x)

    # Add a dropout layer
    x = Dropout(0.5)(x)

    # Add a dense layer with a softmax activation function
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model