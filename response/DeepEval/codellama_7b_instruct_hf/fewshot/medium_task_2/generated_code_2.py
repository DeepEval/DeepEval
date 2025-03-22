import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)
    conv1 = Conv2D(32, (3, 3), activation='relu')(main_path)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    # Define the branch path
    branch_path = Input(shape=input_shape)
    conv3 = Conv2D(64, (5, 5), activation='relu')(branch_path)

    # Combine the main and branch paths
    merged = keras.layers.Concatenate()([pool1, conv3])

    # Flatten and map to a probability distribution across 10 classes
    flatten = Flatten()(merged)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=[main_path, branch_path], outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit([x_train_main, x_train_branch], y_train, epochs=10, validation_data=([x_test_main, x_test_branch], y_test))

    return model