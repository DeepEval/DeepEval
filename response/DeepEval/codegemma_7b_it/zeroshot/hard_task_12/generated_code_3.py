import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, add, Flatten, Dense

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 64))

    # Main path
    x_main = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu')(inputs)
    x_main_1x1 = Conv2D(filters=16, kernel_size=1, padding='same', activation='relu')(x_main)
    x_main_3x3 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(x_main)
    x_main_concat = concatenate([x_main_1x1, x_main_3x3])

    # Branch path
    x_branch = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')(inputs)

    # Combine main and branch paths
    combined = add([x_main_concat, x_branch])

    # Classification layers
    flatten = Flatten()(combined)
    dense_1 = Dense(units=128, activation='relu')(flatten)
    dense_2 = Dense(units=64, activation='relu')(dense_1)
    outputs = Dense(units=10, activation='softmax')(dense_2)

    # Model creation
    model = Model(inputs=inputs, outputs=outputs)

    return model