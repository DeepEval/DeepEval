import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1_2)

    # Branch path
    conv2_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling_branch = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_1)

    # Adding the outputs of both paths
    adding_layer = Add()([max_pooling_main, max_pooling_branch])

    # Flattening the combined output
    flatten_layer = Flatten()(adding_layer)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Constructing the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model