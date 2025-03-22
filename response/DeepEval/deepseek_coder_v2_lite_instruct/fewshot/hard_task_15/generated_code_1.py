import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Reshape, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv_main)
    pool_main = GlobalAveragePooling2D()(conv_main)
    dense_main_1 = Dense(units=64, activation='relu')(pool_main)
    dense_main_2 = Dense(units=32, activation='relu')(dense_main_1)
    weights_main = Reshape((1, 1, 32))(dense_main_2)  # Reshape to match input shape
    weighted_input_main = Multiply()([conv_main, weights_main])  # Element-wise multiplication

    # Branch path
    branch = input_layer  # Directly connected to the input without any modification

    # Adding the outputs from both paths
    added_output = Add()([weighted_input_main, branch])

    # Final fully connected layers
    fc1 = Dense(units=128, activation='relu')(added_output)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model