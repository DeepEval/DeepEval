from keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the four parallel branches
    path1 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    path2 = AveragePooling2D((2, 2), strides=(2, 2))(input_shape)
    path3 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    path3 = Conv2D(64, (3, 3), activation='relu')(path3)
    path3 = Conv2D(64, (3, 3), activation='relu')(path3)
    path4 = Conv2D(64, (1, 1), activation='relu')(input_shape)
    path4 = Conv2D(64, (3, 3), activation='relu')(path4)
    path4 = Conv2D(64, (3, 3), activation='relu')(path4)

    # Concatenate the outputs of the parallel branches
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a dropout layer
    dropout = Dropout(0.5)(flattened)

    # Add a fully connected layer with a softmax activation function
    fc = Dense(10, activation='softmax')(dropout)

    # Create the model
    model = Model(inputs=input_shape, outputs=fc)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model