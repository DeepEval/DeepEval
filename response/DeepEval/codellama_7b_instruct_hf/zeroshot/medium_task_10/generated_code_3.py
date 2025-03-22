import keras
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add
from keras.models import Model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator



def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Convert the labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the VGG16 model
    vgg16 = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    vgg16.trainable = False

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the convolutional layers
    conv_layer1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    conv_layer2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv_layer1)
    conv_layer3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv_layer2)

    # Define the residual layers
    residual_layer1 = Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    residual_layer2 = Conv2D(64, (3, 3), activation="relu", padding="same")(residual_layer1)
    residual_layer3 = Conv2D(128, (3, 3), activation="relu", padding="same")(residual_layer2)

    # Define the addition layers
    add_layer1 = Add()([conv_layer3, residual_layer3])
    add_layer2 = Add()([add_layer1, residual_layer2])
    add_layer3 = Add()([add_layer2, residual_layer1])

    # Define the output layer
    output_layer = Conv2D(10, (3, 3), activation="softmax")(add_layer3)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    return model