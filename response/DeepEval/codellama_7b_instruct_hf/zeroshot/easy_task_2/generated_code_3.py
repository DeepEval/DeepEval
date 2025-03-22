from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Softmax

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)

    # Define the sequential feature extraction layers
    conv1 = Conv2D(32, (3, 3), activation='relu')
    conv2 = Conv2D(64, (3, 3), activation='relu')
    pool1 = MaxPooling2D(pool_size=(2, 2))
    pool2 = MaxPooling2D(pool_size=(2, 2))

    # Define the convolutional layers
    conv3 = Conv2D(128, (3, 3), activation='relu')
    conv4 = Conv2D(128, (3, 3), activation='relu')
    conv5 = Conv2D(128, (3, 3), activation='relu')

    # Define the average pooling layer
    pool3 = MaxPooling2D(pool_size=(2, 2))

    # Define the flatten layer
    flatten = Flatten()

    # Define the fully connected layers
    fc1 = Dense(128, activation='relu')
    fc2 = Dense(64, activation='relu')

    # Define the dropout layers
    dropout1 = Dropout(0.25)
    dropout2 = Dropout(0.5)

    # Define the softmax layer
    softmax = Softmax(1000)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the sequential feature extraction layers
    x = conv1(inputs)
    x = conv2(x)
    x = pool1(x)
    x = pool2(x)

    # Define the convolutional layers
    x = conv3(x)
    x = conv4(x)
    x = conv5(x)

    # Define the average pooling layer
    x = pool3(x)

    # Define the flatten layer
    x = flatten(x)

    # Define the fully connected layers
    x = fc1(x)
    x = dropout1(x)
    x = fc2(x)
    x = dropout2(x)

    # Define the softmax layer
    outputs = softmax(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model