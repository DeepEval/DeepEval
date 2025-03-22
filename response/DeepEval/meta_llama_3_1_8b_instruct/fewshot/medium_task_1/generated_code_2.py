import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)

    # Define the max-pooling layer
    max_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)

    # Add the input layer with the max-pooled features
    adding_layer = Add()([input_layer, max_pool])

    # Define the max-pooling layer again
    max_pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(adding_layer)

    # Flatten the features
    flatten_layer = Flatten()(max_pool_2)

    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)

    # Define the second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model