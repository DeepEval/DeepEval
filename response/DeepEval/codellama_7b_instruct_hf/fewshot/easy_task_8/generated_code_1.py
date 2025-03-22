import keras
from keras.layers import Input, DepthwiseSeparableConv2D, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the depthwise separable convolutional layer
    conv1 = DepthwiseSeparableConv2D(kernel_size=(3, 3), strides=(1, 1), activation='relu')(input_shape)

    # Define the 1x1 convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(conv1)

    # Define the dropout layers
    drop1 = Dropout(0.2)(conv2)
    drop2 = Dropout(0.2)(drop1)

    # Define the max pooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(drop2)

    # Define the flatten layer
    flatten = Flatten()(maxpool)

    # Define the fully connected layer
    fc = Dense(units=128, activation='relu')(flatten)
    fc = Dropout(0.2)(fc)

    # Define the output layer
    output = Dense(units=10, activation='softmax')(fc)

    # Create the model
    model = keras.Model(inputs=input_shape, outputs=output)

    return model