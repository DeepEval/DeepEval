from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, Flatten, Concatenate


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    block_1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    block_1 = GlobalAveragePooling2D()(block_1)
    block_1 = Dense(32, activation='relu')(block_1)
    block_1 = Dense(32, activation='relu')(block_1)
    block_1 = Flatten()(block_1)

    # Define the second block
    block_2 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    block_2 = GlobalAveragePooling2D()(block_2)
    block_2 = Dense(32, activation='relu')(block_2)
    block_2 = Dense(32, activation='relu')(block_2)
    block_2 = Flatten()(block_2)

    # Concatenate the outputs of the two blocks
    concatenated = Concatenate()([block_1, block_2])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a fully connected layer with softmax activation
    output = Dense(10, activation='softmax')(flattened)

    # Define the model
    model = Model(inputs=input_shape, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    # Evaluate the model
    model.evaluate(x_test, y_test)

    return model