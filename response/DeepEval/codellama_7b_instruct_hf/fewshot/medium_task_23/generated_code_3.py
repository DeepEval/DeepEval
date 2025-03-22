import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the paths
    path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_shape)
    path2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), activation='relu')(input_shape)
    path3 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), activation='relu')(input_shape)
    path4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(input_shape)

    # Concatenate the outputs of the paths
    concatenated = Concatenate()([path1, path2, path3, path4])

    # Flatten the concatenated output
    flattened = Flatten()(concatenated)

    # Add a fully connected layer
    fc = Dense(units=128, activation='relu')(flattened)

    # Add a final fully connected layer for classification
    output = Dense(units=10, activation='softmax')(fc)

    # Define the model
    model = Model(inputs=input_shape, outputs=output)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model