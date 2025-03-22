import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_shape)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)

    # Define the second convolutional block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)

    # Define the third convolutional block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv3)

    # Define the fourth convolutional block
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv4)

    # Define the fifth convolutional block
    conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv5)

    # Define the fully connected layers
    flatten = Flatten()(pool5)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=dense2)

    return model