import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)

    # Define the first max pooling layer
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool1)

    # Define the second max pooling layer
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)

    # Define the third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(max_pool2)

    # Define the third max pooling layer
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv3)

    # Define the concatenation layer
    concatenate = Concatenate()([conv1, conv2, conv3, max_pool1, max_pool2, max_pool3])

    # Define the batch normalization layer
    batch_norm = BatchNormalization()(concatenate)

    # Define the flatten layer
    flatten = Flatten()(batch_norm)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten)

    # Define the second fully connected layer
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the output layer
    output = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model