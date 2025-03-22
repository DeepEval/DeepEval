import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(conv1)

    # Branch path
    branch_layer = Input(shape=(28, 28, 1))
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(branch_layer)
    branch_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)

    # Combine main and branch path through an addition operation
    combined_layer = Concatenate()([conv2, branch_pool])

    # Flatten and fully connected layers
    flat_layer = Flatten()(combined_layer)
    dense1 = Dense(units=128, activation='relu')(flat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model