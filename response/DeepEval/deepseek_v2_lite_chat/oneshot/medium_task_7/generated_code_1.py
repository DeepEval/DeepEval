import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional Layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Convolutional Layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Convolutional Layer 3
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv3)
    conv3 = BatchNormalization()(conv3)

    # Separate Convolutional Layer for direct input
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input_layer)
    conv4 = BatchNormalization()(conv4)

    # Concatenate the outputs from all paths
    concat = Concatenate()([conv1, conv2, conv3, conv4])

    # Fully Connected Layer 1
    dense1 = Dense(units=512, activation='relu')(concat)
    dense1 = BatchNormalization()(dense1)

    # Fully Connected Layer 2
    dense2 = Dense(units=256, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model