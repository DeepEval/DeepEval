import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First Feature Extraction Path
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Second Feature Extraction Path
    conv1_7_7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv7_1_7 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(conv1_7_7)

    # Concatenation and 1x1 Convolution
    concat = Concatenate()([conv1_1, conv7_1_7])
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)

    # Branch from Input
    branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Merging Branches
    merged = keras.layers.Add()([conv_1x1, branch])

    # Fully Connected Layers
    flatten_layer = Flatten()(merged)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model