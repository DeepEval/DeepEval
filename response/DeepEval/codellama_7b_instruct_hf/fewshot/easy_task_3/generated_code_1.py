import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense


def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # First convolutional block
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_1)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1_2)

    # Second convolutional block
    conv2_1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_1)
    conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2_2)

    # Flatten the feature maps and add a fully connected layer
    flatten = Flatten()(pool4)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=dense)

    
    return model