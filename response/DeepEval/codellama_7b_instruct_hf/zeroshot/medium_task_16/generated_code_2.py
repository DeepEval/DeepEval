import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the input data
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    inputs = Input(shape=input_shape)
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(inputs)
    x = Conv2D(64, (3, 3), activation="relu")(x[0])
    x = Conv2D(64, (3, 3), activation="relu")(x[1])
    x = Conv2D(64, (3, 3), activation="relu")(x[2])
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)

    # Define the model
    model = keras.models.Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Return the constructed model
    return model