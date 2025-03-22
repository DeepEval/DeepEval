from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Add, Flatten, Dense, Reshape
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # Normalized input image shape
    input_tensor = Input(shape=input_shape)

    # Feature extraction branches
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
    branch2 = Conv2D(64, (1, 1))(branch1)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)
    branch3 = Conv2D(64, (1, 1))(branch1)
    branch3 = Conv2D(64, (3, 3))(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)

    # Concatenate the outputs of the branches
    x = concatenate([branch2, branch3, branch3])

    # Adjust output dimensions to match input image's channel size
    x = Conv2D(3, (1, 1), activation='sigmoid')(x)

    # Main path directly connected to input
    main_path = Conv2D(64, (1, 1), activation='relu')(input_tensor)

    # Fuse the main path and the branch through addition
    model = Add()([main_path, x])

    # Flatten and pass through fully connected layers
    model = Flatten()(model)
    model = Dense(128, activation='relu')(model)
    model = Dense(64, activation='relu')(model)
    model = Dense(10, activation='softmax')(model)  # Assuming you want a multi-class classification

    # Define the model
    model = Model(inputs=input_tensor, outputs=model)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Return the model
    return model