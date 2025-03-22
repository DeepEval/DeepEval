import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dropout, Add, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define input layer
    input_layer = Input(shape=(28, 28, 1))  # MNIST dataset images are 28x28 and grayscale

    # First 1x1 Convolutional layer followed by a Dropout layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    x = Dropout(rate=0.2)(x)

    # Second 1x1 Convolutional layer followed by a Dropout layer
    x = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    # 3x1 Convolutional layer followed by a Dropout layer
    x = Conv2D(filters=32, kernel_size=(3, 1), activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    # 1x3 Convolutional layer followed by a Dropout layer
    x = Conv2D(filters=32, kernel_size=(1, 3), activation='relu')(x)
    x = Dropout(rate=0.2)(x)

    # 1x1 Convolutional layer to restore the number of channels to match the input
    x = Conv2D(filters=1, kernel_size=(1, 1), activation='relu')(x)

    # Combine processed features with the original input via addition
    x = Add()([x, input_layer])

    # Flatten the output
    x = Flatten()(x)

    # Fully connected layer to produce the final probability distribution
    output_layer = Dense(units=10, activation='softmax')(x)  # 10 classes for MNIST

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model