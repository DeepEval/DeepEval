import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_initial = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_initial)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    branch_2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = UpSampling2D(size=(2, 2))(branch_2)
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv_initial)
    branch_3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_3)
    branch_3 = UpSampling2D(size=(2, 2))(branch_3)
    branch_output = Concatenate(axis=-1)([branch_1, branch_2, branch_3])
    conv_main = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_output)

    # Branch path
    conv_branch = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Concatenation and addition
    combined = Concatenate(axis=-1)([conv_main, conv_branch])
    combined = BatchNormalization()(combined)
    combined = Flatten()(combined)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(combined)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model