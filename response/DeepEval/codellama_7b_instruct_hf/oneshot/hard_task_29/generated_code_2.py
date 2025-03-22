import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))

    # Define the first block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch_path = Concatenate()([conv1, conv2])

    # Define the second block
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_path)
    max_pool2 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(max_pool1)
    max_pool3 = MaxPooling2D(pool_size=(8, 8), strides=8, padding='valid')(max_pool2)
    flatten_layer = Flatten()(max_pool3)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model