import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.vgg16 import VGG16

from keras.models import Model
from keras.layers import Input, Dense, Flatten, Lambda, MaxPooling2D, AvgPooling2D, Conv2D, DepthwiseSeparableConv2D
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet_v2 import MobileNetV2


def dl_model():
    
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize the pixel values to be between 0 and 1
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    input_layer = Input(shape=input_shape)

    # Define the convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Define the fully connected layers
    flatten = Flatten()(pool2)
    fc1 = Dense(128, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    return model