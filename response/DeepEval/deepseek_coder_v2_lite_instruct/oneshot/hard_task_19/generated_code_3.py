import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, BatchNormalization, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv2)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Branch path
    global_avg_pool = GlobalAveragePooling2D()(max_pool)
    dense1 = Dense(units=128, activation='relu')(global_avg_pool)
    dense2 = Dense(units=64, activation='relu')(dense1)
    weights = Dense(units=128, activation='softmax')(dense2)
    weights_reshaped = keras.layers.Reshape((1, 1, 128))(weights)

    # Apply weights to the input
    multiplied_input = Multiply()([max_pool, weights_reshaped])

    # Add outputs from both paths
    added_output = Add()([max_pool, multiplied_input])

    # Flatten and add more fully connected layers
    flatten_layer = Flatten()(added_output)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model