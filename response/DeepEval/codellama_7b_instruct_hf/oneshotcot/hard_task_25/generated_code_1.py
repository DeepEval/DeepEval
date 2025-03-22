from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add
from keras.models import Model


def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Conv2D(32, (1, 1), activation='relu')(input_shape)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    main_path = MaxPooling2D((2, 2))(main_path)

    # Define the branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(input_shape)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = MaxPooling2D((2, 2))(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)
    branch_path = Conv2D(32, (3, 3), activation='relu')(branch_path)

    # Concatenate the outputs of the main and branch paths
    output = Concatenate()([main_path, branch_path])
    output = Conv2D(32, (1, 1), activation='relu')(output)
    output = Flatten()(output)
    output = Dense(128, activation='relu')(output)
    output = Dense(10, activation='softmax')(output)

    # Create the model
    model = Model(inputs=input_shape, outputs=output)


    # Load the dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Define the model
    model = dl_model()

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model