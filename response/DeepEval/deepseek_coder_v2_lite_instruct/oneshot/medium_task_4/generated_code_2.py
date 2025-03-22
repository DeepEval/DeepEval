import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def block(input_tensor, filters, kernel_size):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(input_tensor)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        return x

    # Path 1
    path1 = block(input_tensor=input_layer, filters=32, kernel_size=(3, 3))
    path1 = block(input_tensor=path1, filters=64, kernel_size=(3, 3))

    # Path 2
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Concatenate outputs from both paths
    concat = Concatenate()([path1, path2])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model