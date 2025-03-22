from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate
from keras.models import Model



def dl_model():
    
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first block
    x = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(input)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    # Define the second block
    y = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(input)
    y = MaxPooling2D((2, 2))(y)
    y = Dropout(0.2)(y)

    # Concatenate the outputs of the two blocks
    z = Concatenate()([x, y])

    # Flatten the concatenated output
    z = Flatten()(z)

    # Add a fully connected layer
    z = Dense(128, activation='relu')(z)

    # Add a dropout layer
    z = Dropout(0.5)(z)

    # Add a final fully connected layer
    z = Dense(10, activation='softmax')(z)

    # Define the model
    model = Model(inputs=input, outputs=z)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model