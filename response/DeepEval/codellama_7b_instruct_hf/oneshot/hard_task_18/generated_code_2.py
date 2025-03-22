import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
    block_output = Concatenate()([conv1, conv2, max_pooling])

    # Second block
    global_avg_pooling = GlobalAveragePooling2D()(block_output)
    flatten_layer = Flatten()(global_avg_pooling)
    dense1 = Dense(128, activation='relu')(flatten_layer)
    dense2 = Dense(10, activation='softmax')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model