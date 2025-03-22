import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the main path
    main_path = Input(shape=input_shape)

    # First convolutional block
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Second convolutional block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_path)

    # Branch path
    branch_path = Input(shape=input_shape)

    # First convolutional block
    branch_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Dropout(rate=0.2)(branch_path)
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_path)
    branch_path = Dropout(rate=0.2)(branch_path)
    branch_path = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_path)

    # Combine the main and branch paths
    main_path = Add()([main_path, branch_path])

    # Flatten and output
    main_path = Flatten()(main_path)
    main_path = Dense(units=10, activation='softmax')(main_path)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=main_path)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    
    return model