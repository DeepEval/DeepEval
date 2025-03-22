import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Data input
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Model input
    input_layer = Input(shape=(28, 28, 1))
    
    # Pathway 1
    def pathway1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bn1)
        return pool1

    pathway1_output = pathway1(input_layer)

    # Repeated block
    for _ in range(3):
        pathway1_output = pathway1(pathway1_output)

    # Pathway 2
    def pathway2(input_tensor):
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn2 = BatchNormalization()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bn2)
        return pool2

    pathway2_output = pathway2(input_layer)

    # Concatenate and Flatten
    concat_layer = Concatenate()([pathway1_output, pathway2_output])
    flatten_layer = Flatten()(concat_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model