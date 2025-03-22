import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First block: Convolution followed by MaxPooling
    conv1 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Second block: Convolution followed by MaxPooling
    conv2 = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Ensure the last convolution output dimension matches the channel dimension of the input
    conv2_adjusted = Conv2D(filters=1, kernel_size=(1, 1), padding='same')(max_pooling2)

    # Combine the input and outputs from both blocks through addition
    added = Add()([input_layer, conv2_adjusted])

    # Flatten the output
    flatten_layer = Flatten()(added)

    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model