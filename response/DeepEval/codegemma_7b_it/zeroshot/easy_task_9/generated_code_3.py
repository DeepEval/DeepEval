import keras
from keras.models import Sequential
from keras.layers import Conv2D, DepthwiseConv2D, Activation, BatchNormalization, Add, Flatten, Dense

def dl_model():

    # Create the model
    model = Sequential()

    # Add the 1x1 convolutional layer to increase dimensionality
    model.add(Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=1))

    # Add the 3x3 depthwise separable convolutional layer for feature extraction
    model.add(DepthwiseConv2D(filters=32, kernel_size=(3, 3), padding='same', strides=1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # Add the 1x1 convolutional layer to reduce dimensionality and maintain a convolutional stride of 1
    model.add(Conv2D(filters=32, kernel_size=(1, 1), padding='same', strides=1))

    # Add the shortcut connection
    model.add(Add()([model.get_layer(index=0).output, model.output]))
    model.add(Activation('relu'))

    # Flatten the output and add a fully connected layer for classification
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    return model

# Compile the model
model = dl_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])