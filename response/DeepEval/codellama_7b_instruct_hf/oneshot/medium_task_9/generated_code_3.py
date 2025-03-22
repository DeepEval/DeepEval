import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv1 = Conv2D(16, (3, 3), activation='relu')(input_layer)

    # Main structure of the model
    basic_block = Conv2D(32, (3, 3), activation='relu')(conv1)
    basic_block = BatchNormalization()(basic_block)
    basic_block = MaxPooling2D(pool_size=(2, 2))(basic_block)

    # Feature fusion
    branch = Conv2D(16, (3, 3), activation='relu')(basic_block)
    branch = BatchNormalization()(branch)
    branch = MaxPooling2D(pool_size=(2, 2))(branch)
    combined = Concatenate()([basic_block, branch])

    # Final convolutional layer
    conv2 = Conv2D(64, (3, 3), activation='relu')(combined)

    # Average pooling layer
    pool = AveragePooling2D(pool_size=(2, 2))(conv2)

    # Flatten layer
    flatten = Flatten()(pool)

    # Fully connected layers
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model