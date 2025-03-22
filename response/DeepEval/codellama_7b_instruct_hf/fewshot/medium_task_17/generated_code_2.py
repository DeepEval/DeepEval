import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Reshape input tensor into three groups
    reshaped_input = input_layer.reshape((32, 32, 3, 1))

    # Swap third and fourth dimensions to enable channel shuffling
    shuffled_input = keras.layers.Permute((3, 1, 2))(reshaped_input)

    # Reshape back to original input shape
    reshaped_input = shuffled_input.reshape((32, 32, 1, 3))

    # Convolutional layers
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_input)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv1)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(conv2)

    # Merge outputs from both paths
    added_layer = Add()([conv1, max_pool1, conv2, max_pool2])

    # Flatten output and pass through fully connected layer for classification
    flattened_layer = Flatten()(added_layer)
    output_layer = Dense(units=10, activation='softmax')(flattened_layer)

    # Define and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model