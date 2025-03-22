import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution for dimensionality reduction
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by 5x5 convolution
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Branch 4: 3x3 max pooling followed by 1x1 convolution
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)

    # Concatenate branch outputs
    concat_output = Concatenate()([branch1, branch2, branch3, branch4])

    # Batch normalization
    bn_output = BatchNormalization()(concat_output)

    # Flatten
    flatten_output = Flatten()(bn_output)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_output)
    dense2 = Dense(units=128, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model