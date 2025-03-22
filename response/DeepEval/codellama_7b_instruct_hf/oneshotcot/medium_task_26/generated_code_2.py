from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer with shape (32, 32, 64)
    input_layer = Input(shape=(32, 32, 64))

    # Add a 1x1 convolutional layer to compress the input channels
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add two parallel convolutional layers with 1x1 and 3x3 convolutions
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Concatenate the results of the two convolutional layers
    concatenated_output = Concatenate()([conv2, conv3])

    # Flatten the output feature map
    flattened_output = Flatten()(concatenated_output)

    # Add two fully connected layers to produce the classification results
    dense1 = Dense(units=128, activation='relu')(flattened_output)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model