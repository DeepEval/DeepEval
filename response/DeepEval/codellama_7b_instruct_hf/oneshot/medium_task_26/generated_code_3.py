import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 64)

    # Define the first layer, a 1x1 convolutional layer
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')

    # Define the second layer, a 3x3 convolutional layer
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    # Define the third layer, a 5x5 convolutional layer
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')

    # Define the fourth layer, a max pooling layer
    maxpool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    # Define the concatenation layer
    concatenate = Concatenate()

    # Define the batch normalization layer
    batch_norm = BatchNormalization()

    # Define the flatten layer
    flatten = Flatten()

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')

    # Define the second fully connected layer
    dense2 = Dense(units=64, activation='relu')

    # Define the output layer
    output_layer = Dense(units=10, activation='softmax')

    # Create the input layer
    input_layer = Input(shape=input_shape)

    # Connect the input layer to the first convolutional layer
    conv1_output = conv1(input_layer)

    # Connect the first convolutional layer to the second convolutional layer
    conv2_output = conv2(conv1_output)

    # Connect the second convolutional layer to the third convolutional layer
    conv3_output = conv3(conv2_output)

    # Connect the third convolutional layer to the concatenation layer
    concatenate_output = concatenate([conv1_output, conv2_output, conv3_output])

    # Connect the concatenation layer to the batch normalization layer
    batch_norm_output = batch_norm(concatenate_output)

    # Connect the batch normalization layer to the flatten layer
    flatten_output = flatten(batch_norm_output)

    # Connect the flatten layer to the first fully connected layer
    dense1_output = dense1(flatten_output)

    # Connect the first fully connected layer to the second fully connected layer
    dense2_output = dense2(dense1_output)

    # Connect the second fully connected layer to the output layer
    output_layer_output = output_layer(dense2_output)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer_output)

    return model