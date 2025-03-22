from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Define the first specialized block
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='sigmoid')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    max_pooling_1 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(conv)
    max_pooling_2 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='valid')(conv)
    max_pooling_3 = MaxPooling2D(pool_size=(16, 16), strides=16, padding='valid')(conv)

    # Flatten the output of each max pooling layer
    flatten_1 = Flatten()(max_pooling)
    flatten_2 = Flatten()(max_pooling_1)
    flatten_3 = Flatten()(max_pooling_2)
    flatten_4 = Flatten()(max_pooling_3)

    # Concatenate the flattened outputs
    concatenated = Concatenate()([flatten_1, flatten_2, flatten_3, flatten_4])

    # Define the second specialized block
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concatenated)
    conv_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concatenated)
    conv_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concatenated)
    max_pooling = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(concatenated)

    # Flatten the output of each convolutional layer and concatenate them
    flatten_1 = Flatten()(conv)
    flatten_2 = Flatten()(conv_1)
    flatten_3 = Flatten()(conv_2)
    flatten_4 = Flatten()(max_pooling)
    concatenated = Concatenate()([flatten_1, flatten_2, flatten_3, flatten_4])

    # Add batch normalization and flatten the output
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Add a fully connected layer and a softmax output
    fc_1 = Dense(units=128, activation='relu')(flattened)
    fc_2 = Dense(units=64, activation='relu')(fc_1)
    output_layer = Dense(units=10, activation='softmax')(fc_2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Train the model on the MNIST dataset
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model