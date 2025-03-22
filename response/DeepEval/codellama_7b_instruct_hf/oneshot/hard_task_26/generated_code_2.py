import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model



def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = Input(shape=input_shape)

    # Apply a 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Apply three branches
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv1)

    # Concatenate the outputs from all branches
    concat = Concatenate()([branch1, branch2, branch3])

    # Apply a 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Apply batch normalization
    batch_norm = BatchNormalization()(conv2)

    # Apply flatten layer
    flatten = Flatten()(batch_norm)

    # Apply two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = Model(inputs=main_path, outputs=dense2)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model