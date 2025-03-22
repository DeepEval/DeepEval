from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet
from keras.applications.vgg import VGG

def dl_model():
    # Load the MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize the inputs from 0-255 to 0-1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the depthwise separable convolutional layer
    sep_conv_layer = Sequential()
    sep_conv_layer.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
    sep_conv_layer.add(DepthwiseSeparableConv2D(32, (3, 3), padding="same", activation="relu"))
    sep_conv_layer.add(MaxPooling2D((2, 2)))

    # Define the 1x1 convolutional layer
    conv_layer = Sequential()
    conv_layer.add(Conv2D(64, (1, 1), activation="relu"))
    conv_layer.add(MaxPooling2D((2, 2)))

    # Define the dropout layer
    dropout_layer = Sequential()
    dropout_layer.add(Dropout(0.2))

    # Define the fully connected layer
    fc_layer = Sequential()
    fc_layer.add(Flatten())
    fc_layer.add(Dense(128, activation="relu"))
    fc_layer.add(Dropout(0.2))
    fc_layer.add(Dense(10, activation="softmax"))

    # Define the model
    model = Sequential()
    model.add(sep_conv_layer)
    model.add(conv_layer)
    model.add(dropout_layer)
    model.add(fc_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model