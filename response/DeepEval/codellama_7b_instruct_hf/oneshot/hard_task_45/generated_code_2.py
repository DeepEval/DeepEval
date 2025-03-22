import keras
from keras.layers import Input, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import CIFAR10
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

def dl_model():
    # Load the CIFAR-10 dataset
    (X_train, y_train), (X_test, y_test) = CIFAR10.load_data()

    # Normalize the pixel values to be between 0 and 1
    X_train = X_train.astype("float32") / 255
    X_test = X_test.astype("float32") / 255

    # Convert the labels to categorical values
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the model architecture
    input_layer = Input(shape=input_shape)

    # Block 1: split the input into three groups and extract features using depthwise separable convolutional layers
    group1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    group2 = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    group3 = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)

    conv1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    conv3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

    # Concatenate the outputs from the three groups
    output_tensor = Concatenate()([conv1, conv2, conv3])

    # Block 2: feature extraction using multiple branches
    branch1 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch2 = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch4 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    branch5 = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(output_tensor)

    # Concatenate the outputs from all branches
    output_tensor = Concatenate()([branch1, branch2, branch3, branch4, branch5])

    # Batch normalization and flatten the output
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with the Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model