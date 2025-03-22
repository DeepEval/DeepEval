# Import necessary packages
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.layers import LeakyReLU, Add, concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.initializers import he_normal
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.optimizers import Adam

def dl_model():
    """
    This function constructs a deep learning model using Keras' Functional API for image classification on the MNIST dataset.
    
    The model architecture features a specialized block designed to capture local features, followed by average pooling and dropout layers. 
    Two consecutive blocks are applied, then global average pooling, flattening, and a fully connected layer produce the final classification output.
    
    :return: The constructed deep learning model.
    """

    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape input data
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Normalize input data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert labels to categorical
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    # Define input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define block to capture local features
    def local_feature_block(x):
        x = Conv2D(64, (3, 3), padding='same', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(64, (1, 1), padding='same', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Conv2D(64, (1, 1), padding='same', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)
        x = Dropout(0.2)(x)
        return x

    # Apply two consecutive local feature blocks
    x = local_feature_block(input_layer)
    x = local_feature_block(x)

    # Apply global average pooling layer
    x = GlobalAveragePooling2D()(x)

    # Apply flattening layer
    x = Flatten()(x)

    # Define fully connected layer for classification
    x = Dense(64, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    output_layer = Dense(10, activation='softmax', kernel_initializer=he_normal(), kernel_regularizer=l2(0.01))(x)

    # Define and compile model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model