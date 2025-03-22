import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second 1x1 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the 3x1 convolutional layer
    conv3 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define the 1x3 convolutional layer
    conv4 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3)

    # Define the dropout layer for each convolutional layer
    drop1 = Dropout(rate=0.2)(conv1)
    drop2 = Dropout(rate=0.2)(conv2)
    drop3 = Dropout(rate=0.2)(conv3)
    drop4 = Dropout(rate=0.2)(conv4)

    # Define the concatenate layer
    concat = Concatenate()([drop1, drop2, drop3, drop4])

    # Define the batch normalization layer
    batch_norm = BatchNormalization()(concat)

    # Define the flatten layer
    flatten = Flatten()(batch_norm)

    # Define the fully connected layer
    dense = Dense(units=128, activation='relu')(flatten)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model