import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model
    input_layer = Input(shape=input_shape)

    # Main path
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    branch2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    branch3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(conv1)

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)

    # Add the outputs of the branches
    adding_layer = Add()([branch1, branch2, branch3])

    # Flatten the output
    flatten_layer = Flatten()(adding_layer)

    # Dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model