# Import necessary packages
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense
from keras.models import Model
from keras.layers import BatchNormalization, Activation
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras import backend as K
import tensorflow as tf

def dl_model():
    # Define the input shape of the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the inputs
    inputs = Input(shape=input_shape)

    # Extract features at multiple scales using 1x1, 3x3, and 5x5 convolutions
    # followed by 3x3 max pooling
    conv1_1 = Conv2D(16, (1, 1), activation='relu')(inputs)
    conv1_1 = BatchNormalization()(conv1_1)

    conv1_2 = Conv2D(16, (3, 3), activation='relu')(inputs)
    conv1_2 = BatchNormalization()(conv1_2)

    conv1_3 = Conv2D(16, (5, 5), activation='relu')(inputs)
    conv1_3 = BatchNormalization()(conv1_3)

    pool1 = MaxPooling2D(pool_size=(3, 3))(inputs)

    # Concatenate the features
    concat1 = concatenate([conv1_1, conv1_2, conv1_3, pool1])

    # Use a 3x3 max pooling operation to reduce the spatial dimensions
    pool2 = MaxPooling2D(pool_size=(3, 3))(concat1)

    # Use a convolutional layer with a 3x3 kernel to extract features
    conv2 = Conv2D(32, (3, 3), activation='relu')(pool2)
    conv2 = BatchNormalization()(conv2)

    # Use a 3x3 max pooling operation to reduce the spatial dimensions
    pool3 = MaxPooling2D(pool_size=(3, 3))(conv2)

    # Use a convolutional layer with a 3x3 kernel to extract features
    conv3 = Conv2D(64, (3, 3), activation='relu')(pool3)
    conv3 = BatchNormalization()(conv3)

    # Use a 3x3 max pooling operation to reduce the spatial dimensions
    pool4 = MaxPooling2D(pool_size=(3, 3))(conv3)

    # Flatten the output to prepare it for the fully connected layers
    flat = Flatten()(pool4)

    # Use two fully connected layers to output classification results
    fc1 = Dense(128, activation='relu')(flat)
    fc1 = BatchNormalization()(fc1)

    outputs = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Test the function
model = dl_model()
model.summary()