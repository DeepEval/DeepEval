import keras
from keras.layers import Input, Dense, Flatten, Dropout, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Concatenate


def dl_model():
    
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the first block
    block1 = Input(shape=input_shape)
    block1 = MaxPooling2D(pool_size=(1, 1))(block1)
    block1 = MaxPooling2D(pool_size=(2, 2))(block1)
    block1 = MaxPooling2D(pool_size=(4, 4))(block1)
    block1 = Flatten()(block1)

    # Define the second block
    block2 = Input(shape=input_shape)
    block2 = Conv2D(32, (1, 1), activation='relu')(block2)
    block2 = Conv2D(64, (3, 3), activation='relu')(block2)
    block2 = Conv2D(128, (5, 5), activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(3, 3))(block2)
    block2 = Flatten()(block2)

    # Concatenate the outputs of the two blocks
    concatenated = Concatenate()([block1, block2])

    # Add a dropout layer
    concatenated = Dropout(0.2)(concatenated)

    # Add a fully connected layer
    concatenated = Dense(128, activation='relu')(concatenated)
    concatenated = Dropout(0.5)(concatenated)

    # Add a classification layer
    concatenated = Dense(10, activation='softmax')(concatenated)

    # Create the model
    model = keras.Model(inputs=block1, outputs=concatenated)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Print the model summary
    print(model.summary())

    return model