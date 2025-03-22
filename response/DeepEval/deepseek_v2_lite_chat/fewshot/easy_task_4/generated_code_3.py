import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Convolutional layers followed by max pooling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Flatten the output
    flatten = Flatten()(pool3)

    # Fully connected layer
    fc1 = Dense(units=128, activation='relu')(flatten)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(fc1)

    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model