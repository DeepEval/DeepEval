import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.vgg16 import VGG16

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the four parallel branches
    path_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    path_3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    path_4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Define the concatenation layer
    concatenate_layer = Concatenate(axis=-1)([path_1, path_2, path_3, path_4])

    # Define the batch normalization and flatten layers
    bath_norm = BatchNormalization()(concatenate_layer)
    flatten_layer = Flatten()(bath_norm)

    # Define the fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model