from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model


def dl_model():

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add two convolutional layers with max-pooling
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    # Add the output of the convolutional layers to the input layer
    merged = Add()([input_layer, pool1])

    # Flatten the merged output
    flattened = Flatten()(merged)

    # Add two fully connected layers for classification
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model