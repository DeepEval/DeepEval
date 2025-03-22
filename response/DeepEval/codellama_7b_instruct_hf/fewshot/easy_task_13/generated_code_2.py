import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

 和 return model
def dl_model():
    # Define the input shape for the model
    input_shape = (28, 28, 1)

    # Define the first convolutional layer
    conv1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_shape)

    # Define the second convolutional layer
    conv2 = Conv2D(32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the third convolutional layer
    conv3 = Conv2D(32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv2)

    # Define the dropout layer
    dropout = Dropout(0.2)(conv3)

    # Define the flattening layer
    flatten = Flatten()(dropout)

    # Define the fully connected layer
    dense = Dense(128, activation='relu')(flatten)

    # Define the output layer
    output = Dense(10, activation='softmax')(dense)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output)

    return model