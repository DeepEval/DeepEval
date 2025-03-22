import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AvgPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    block1 = Concatenate()([pool1, pool2])

    # Second block
    avg_pool = AvgPooling2D(pool_size=(8, 8), strides=8, padding='valid')(block1)
    flat = Flatten()(avg_pool)
    dense1 = Dense(units=64, activation='relu')(flat)
    dense2 = Dense(units=32, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output)
    return model