import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the inputs
    input_layer = Input(shape=(32, 32, 3))

    # Define the branches
    branch_1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    branch_2 = Conv2D(32, (1, 1), activation='relu')(input_layer)
    branch_2 = Conv2D(32, (3, 3), activation='relu')(branch_2)
    branch_3 = MaxPooling2D(pool_size=(2, 2))(input_layer)

    # Concatenate the outputs from the branches
    concatenated = keras.layers.Concatenate()([branch_1, branch_2, branch_3])

    # Flatten the concatenated outputs
    flattened = Flatten()(concatenated)

    # Pass the flattened outputs through two fully connected layers
    fc_1 = Dense(64, activation='relu')(flattened)
    fc_2 = Dense(10, activation='softmax')(fc_1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc_2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the summary of the model
    print(model.summary())


    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    return model