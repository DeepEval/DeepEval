import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Add, BatchNormalization, Flatten, Dense, Multiply, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    add1 = Add()([input_layer, conv2])  # Element-wise addition
    avg_pool = AveragePooling2D(pool_size=(2, 2))(add1)

    # Second block
    global_avg_pool = GlobalAveragePooling2D()(conv2)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    reshape_weights = Dense(units=conv2.shape[1] * conv2.shape[2] * conv2.shape[3], activation='softmax')(dense2)
    reshape_weights = keras.layers.Reshape((conv2.shape[1], conv2.shape[2], conv2.shape[3]))(reshape_weights)
    multiply_weights = Multiply()([conv2, reshape_weights])
    flatten_layer = Flatten()(multiply_weights)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Example usage:
# model = dl_model()
# model.summary()