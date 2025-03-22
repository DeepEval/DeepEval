import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first stage of convolutions and max pooling
    x = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the second stage of convolutions and max pooling
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Define the skip connections
    skip1 = Conv2D(32, (3, 3), activation='relu')(x)
    skip2 = Conv2D(64, (3, 3), activation='relu')(x)

    # Define the upsampling layers
    x = UpSampling2D((2, 2))(skip1)
    x = Concatenate()([x, skip2])

    # Define the final convolutional layer
    x = Conv2D(128, (3, 3), activation='relu')(x)

    # Define the output layer
    x = Conv2D(10, (1, 1), activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_shape, outputs=x)

    return model