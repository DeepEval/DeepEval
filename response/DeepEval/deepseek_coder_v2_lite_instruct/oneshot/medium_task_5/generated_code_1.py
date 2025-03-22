import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    def main_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x

    main_output = main_path(input_layer)

    # Branch path
    def branch_path(x):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
        return x

    branch_output = branch_path(input_layer)

    # Addition of main and branch paths
    combined_output = Add()([main_output, branch_output])

    # Flatten the output
    flatten_layer = Flatten()(combined_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=dense2)

    return model