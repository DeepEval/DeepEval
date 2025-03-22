import keras
from keras.layers import Input, MaxPooling2D, Flatten, Concatenate, Dense, Reshape, Conv2D

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Block 1: Multiple Max Pooling Layers
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(input_layer)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(input_layer)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(input_layer)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(pool2)
    flat3 = Flatten()(pool3)

    concat1 = Concatenate()([flat1, flat2, flat3])

    # Fully Connected Layer and Reshape
    dense1 = Dense(units=128, activation='relu')(concat1)
    reshape_layer = Reshape((1, 128))(dense1)

    # Block 2: Feature Extraction Branches
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(reshape_layer)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(reshape_layer)

    concat2 = Concatenate()([conv1, conv2, conv3, pool4])

    # Final Layers
    flatten_layer = Flatten()(concat2)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model