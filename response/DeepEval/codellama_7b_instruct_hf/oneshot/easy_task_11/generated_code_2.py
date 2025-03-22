import keras
from keras.layers import Input, AveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the average pooling layer
    pool_layer = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # Define the 1x1 convolutional layer
    conv_layer = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(pool_layer)

    # Define the flatten layer
    flatten_layer = Flatten()(conv_layer)

    # Define the first fully connected layer
    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)

    # Define the dropout layer
    dropout_layer = Dropout(rate=0.2)(dense_layer1)

    # Define the second fully connected layer
    dense_layer2 = Dense(units=64, activation='relu')(dropout_layer)

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model