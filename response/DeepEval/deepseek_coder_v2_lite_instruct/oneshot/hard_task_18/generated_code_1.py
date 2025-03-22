import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, GlobalAveragePooling2D, Dense, Multiply, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    add1 = Add()([conv2, input_layer])  # Add the input to the output of the second conv layer
    avg_pool1 = AveragePooling2D(pool_size=(2, 2), strides=2)(add1)

    # Second block
    global_avg_pool = GlobalAveragePooling2D()(avg_pool1)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)

    # Reshape weights to match the number of channels in the input
    reshape_weights = Dense(units=32, activation='relu')(dense2)
    reshape_weights = keras.backend.expand_dims(reshape_weights, -1)
    reshape_weights = keras.backend.expand_dims(reshape_weights, -1)

    # Element-wise multiplication of the input with the refined weights
    weighted_input = Multiply()([avg_pool1, reshape_weights])
    flatten_layer = Flatten()(weighted_input)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model