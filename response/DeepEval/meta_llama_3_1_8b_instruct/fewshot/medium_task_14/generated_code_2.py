import keras
from keras.layers import Input, Conv2D, BatchNormalization, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        activation = keras.layers.ReLU()(batch_norm)
        return activation

    block1 = block(input_layer)
    block2 = block(block1)
    block3 = block(block2)

    parallel_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    batch_norm = BatchNormalization()(parallel_branch)
    activation = keras.layers.ReLU()(batch_norm)

    adding_layer = Add()([block3, activation])

    flatten_layer = Flatten()(adding_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

def main():
    model = dl_model()
    model.summary()
    return model

if __name__ == "__main__":
    model = main()