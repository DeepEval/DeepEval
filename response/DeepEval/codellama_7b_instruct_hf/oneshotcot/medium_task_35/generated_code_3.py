import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=input_shape)(input_shape)
    max_pooling1 = MaxPooling2D((2, 2))(conv1)

    # Define the second convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D((2, 2))(conv2)

    # Define the third convolutional layer
    conv3 = Conv2D(128, (3, 3), activation='relu')(max_pooling2)
    max_pooling3 = MaxPooling2D((2, 2))(conv3)

    # Define the skip connection layers
    skip_connection1 = conv1(max_pooling2)
    skip_connection2 = conv2(max_pooling3)

    # Define the concatenation layer
    concatenation = Concatenate()([skip_connection1, skip_connection2])

    # Define the batch normalization layer
    batch_normalization = BatchNormalization()(concatenation)

    # Define the flatten layer
    flatten = Flatten()(batch_normalization)

    # Define the dense layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=output_layer)

    return model