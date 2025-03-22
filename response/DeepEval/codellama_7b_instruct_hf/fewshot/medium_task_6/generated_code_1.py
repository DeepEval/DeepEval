import keras
from keras.layers import Conv2D, BatchNormalization, ReLU, Flatten, Dense


def dl_model():
    # Define the model architecture
    input_layer = keras.Input(shape=(32, 32, 3))

    # Initial convolution
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Parallel blocks
    block1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    block1 = BatchNormalization()(block1)
    block1 = ReLU()(block1)

    block2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    block2 = BatchNormalization()(block2)
    block2 = ReLU()(block2)

    block3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    block3 = BatchNormalization()(block3)
    block3 = ReLU()(block3)

    # Add the output of each block
    added_blocks = keras.layers.Add()([block1, block2, block3])

    # Flatten the output
    flattened_output = Flatten()(added_blocks)

    # Dense layers
    dense1 = Dense(64, activation='relu')(flattened_output)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=dense3)

    # Compile the model with a suitable optimizer and loss function
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    return model