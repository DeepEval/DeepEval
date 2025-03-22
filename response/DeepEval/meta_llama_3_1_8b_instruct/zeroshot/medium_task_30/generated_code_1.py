# Import necessary packages
from keras.models import Model
from keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from keras.layers import Dropout
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam

def dl_model():
    """
    This function constructs a deep learning model using the Functional APIs of Keras 
    for image classification on the CIFAR-10 dataset.
    
    The model consists of three average pooling layers with pooling windows and strides 
    of 1x1, 2x2, and 4x4, allowing it to capture spatial information at different scales.
    The outputs of these pooling layers are flattened into one-dimensional vectors and 
    concatenated. After concatenation, the fused features are further flattened and 
    processed through two fully connected layers to generate the final classification results.
    """

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the input layer
    inputs = Input(shape=input_shape)

    # Define the convolutional and pooling layers
    x = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(inputs)
    x = AveragePooling2D(pool_size=(1, 1), strides=1)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=4)(x)

    # Flatten the pooling layers' outputs
    x = Flatten()(x)

    # Concatenate the flattened outputs
    x = Concatenate()([x])

    # Define the fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    return model