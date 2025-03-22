import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the three branches of the model
    branch_1 = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch_2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_3 = Conv2D(32, (1, 7), activation='relu')(input_shape)
    branch_4 = Conv2D(32, (7, 1), activation='relu')(input_shape)
    branch_5 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_6 = MaxPooling2D((2, 2))(branch_5)

    # Dropout layers to mitigate overfitting
    branch_1 = Dropout(0.2)(branch_1)
    branch_2 = Dropout(0.2)(branch_2)
    branch_3 = Dropout(0.2)(branch_3)
    branch_4 = Dropout(0.2)(branch_4)
    branch_5 = Dropout(0.2)(branch_5)
    branch_6 = Dropout(0.2)(branch_6)

    # Concatenate the outputs from all branches
    concatenated_output = keras.layers.concatenate([branch_1, branch_2, branch_3, branch_4, branch_5, branch_6])

    # Flatten the concatenated output
    flattened_output = Flatten()(concatenated_output)

    # Add a fully connected layer with 128 units
    flattened_output = Dense(128, activation='relu')(flattened_output)

    # Add a dropout layer
    flattened_output = Dropout(0.5)(flattened_output)

    # Add another fully connected layer with 10 units (for the 10 classes in the CIFAR-10 dataset)
    output = Dense(10, activation='softmax')(flattened_output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model