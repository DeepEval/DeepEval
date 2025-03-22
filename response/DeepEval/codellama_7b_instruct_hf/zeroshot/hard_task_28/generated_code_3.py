import keras
from keras.layers import Conv2D, BatchNormalization, Add, Flatten, Dense
from keras.models import Model
from keras.applications import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the main path
    main_path = Conv2D(64, (7, 7), padding="same", activation="relu")(X_train)
    main_path = BatchNormalization()(main_path)
    main_path = Conv2D(64, (1, 1), padding="same", activation="relu")(main_path)
    main_path = Conv2D(64, (1, 1), padding="same", activation="relu")(main_path)
    main_path = Flatten()(main_path)

    # Define the branch path
    branch_path = X_train

    # Combine the main and branch paths
    combined_path = Add()([main_path, branch_path])

    # Define the final layers
    final_path = Dense(10, activation="softmax")(combined_path)

    # Create the model
    model = Model(inputs=X_train, outputs=final_path)

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model