from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

 和 return model
def dl_model(): 

    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the two branches
    branch_1 = Conv2D(32, (3, 3), activation='relu')(input_shape)
    branch_2 = Conv2D(32, (3, 3), activation='relu')(input_shape)

    # Define the first block
    block_1 = Conv2D(64, (3, 3), activation='relu')(branch_1)
    block_1 = MaxPooling2D((2, 2))(block_1)

    # Define the second block
    block_2 = Conv2D(64, (3, 3), activation='relu')(branch_2)
    block_2 = MaxPooling2D((2, 2))(block_2)

    # Add the outputs from the two blocks
    added_blocks = Add()([block_1, block_2])

    # Flatten the output
    flattened_output = Flatten()(added_blocks)

    # Define the final classification layer
    output_layer = Dense(10, activation='softmax')(flattened_output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Print the summary of the model
    print(model.summary())


    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


    # Evaluate the model on the test dataset
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test loss: {test_loss:.3f}, Test accuracy: {test_acc:.3f}')

    return model