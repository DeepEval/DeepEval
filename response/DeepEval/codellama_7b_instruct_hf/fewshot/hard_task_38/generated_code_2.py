from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Concatenate, Dense

def dl_model():
    # Define the input shape and the number of classes
    input_shape = (28, 28, 1)
    num_classes = 10

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the two processing pathways
    pathway1 = inputs
    pathway2 = inputs

    # Define the repeated block structure for each pathway
    for i in range(3):
        # Apply batch normalization and ReLU activation to the input
        pathway1 = BatchNormalization()(pathway1)
        pathway1 = Activation('relu')(pathway1)

        # Apply a 3x3 convolutional layer to extract features
        pathway1 = Conv2D(32, (3, 3), padding='same')(pathway1)

        # Merge the original input with the new features along the channel dimension
        pathway1 = Concatenate(axis=3)([inputs, pathway1])

    # Define the second processing pathway
    for i in range(3):
        # Apply batch normalization and ReLU activation to the input
        pathway2 = BatchNormalization()(pathway2)
        pathway2 = Activation('relu')(pathway2)

        # Apply a 3x3 convolutional layer to extract features
        pathway2 = Conv2D(32, (3, 3), padding='same')(pathway2)

        # Merge the original input with the new features along the channel dimension
        pathway2 = Concatenate(axis=3)([inputs, pathway2])

    # Merge the outputs from both pathways through concatenation
    outputs = Concatenate(axis=3)([pathway1, pathway2])

    # Apply two fully connected layers for classification
    outputs = Dense(64, activation='relu')(outputs)
    outputs = Dense(num_classes, activation='softmax')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model