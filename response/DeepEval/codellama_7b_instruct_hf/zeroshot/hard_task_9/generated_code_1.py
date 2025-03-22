from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.applications.cifar10 import Cifar10
import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten
from keras.applications.vgg16 import VGG16


def dl_model():

    # Load the Cifar10 dataset
    (X_train, y_train), (X_test, y_test) = Cifar10.load_data()

    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the first feature extraction branch
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_shape)
    branch1 = MaxPooling2D(pool_size=(2, 2))(branch1)

    # Define the second feature extraction branch
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_shape)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=(2, 2))(branch2)

    # Define the third feature extraction branch
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_shape)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = MaxPooling2D(pool_size=(2, 2))(branch3)

    # Concatenate the outputs of the feature extraction branches
    branch_outputs = Concatenate()([branch1, branch2, branch3])

    # Flatten the concatenated output
    branch_outputs = Flatten()(branch_outputs)

    # Define the classification layer
    classification = Dense(10, activation='softmax')(branch_outputs)

    # Create the model
    model = Model(inputs=input_shape, outputs=classification)

    return model