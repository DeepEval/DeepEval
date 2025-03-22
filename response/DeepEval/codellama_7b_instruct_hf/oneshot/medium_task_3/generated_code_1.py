import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the max pooling layer
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)

    # Define the max pooling layer
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Define the third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)

    # Define the max pooling layer
    max_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)

    # Define the fourth convolutional layer
    conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling3)

    # Define the max pooling layer
    max_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv4)

    # Define the concatenation layer
    concatenate = Concatenate()([max_pooling1, max_pooling2, max_pooling3, max_pooling4])

    # Define the batch normalization layer
    batch_normalization = BatchNormalization()(concatenate)

    # Define the flatten layer
    flatten = Flatten()(batch_normalization)

    # Define the fully connected layer
    dense = Dense(units=10, activation='softmax')(flatten)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model