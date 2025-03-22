import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv_main)
    global_avg_pool_main = GlobalAveragePooling2D()(conv_main)
    dense1_main = Dense(units=64, activation='relu')(global_avg_pool_main)
    dense2_main = Dense(units=32, activation='relu')(dense1_main)
    reshaped_weights_main = Dense(units=32*32*3, activation='sigmoid')(dense2_main)
    reshaped_weights_main = keras.backend.reshape(reshaped_weights_main, (-1, 32, 32, 3))
    weighted_input_main = Multiply()([input_layer, reshaped_weights_main])

    # Branch path
    weighted_input_branch = input_layer

    # Combine outputs
    combined = Add()([weighted_input_main, weighted_input_branch])

    # Final layers
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model