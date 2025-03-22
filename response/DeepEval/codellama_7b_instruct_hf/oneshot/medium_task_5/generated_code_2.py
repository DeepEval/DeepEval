import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the main path
    main_path = VGG16(include_top=False, input_shape=input_shape, weights=None)

    # Define the branch path
    branch_path = VGG16(include_top=False, input_shape=input_shape, weights=None)

    # Freeze the layers in the main path
    for layer in main_path.layers:
        layer.trainable = False

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the first block in the main path
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv2)

    # Define the second block in the main path
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D((2, 2))(conv4)

    # Define the first block in the branch path
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D((2, 2))(conv6)

    # Define the concatenation layer
    concatenate = Concatenate()([pool1, pool2, pool3])

    # Define the batch normalization layer
    batch_normalization = BatchNormalization()(concatenate)

    # Define the flattening layer
    flatten = Flatten()(batch_normalization)

    # Define the fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model