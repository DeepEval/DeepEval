import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Convolutional layer and max pooling layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Additional convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(max_pooling1)

    # Flatten the feature maps
    flatten = Flatten()(conv2)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model