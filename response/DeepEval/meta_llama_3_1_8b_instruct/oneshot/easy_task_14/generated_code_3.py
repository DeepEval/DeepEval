import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 dataset has images of size 32x32 with 3 color channels
    global_average_pool = GlobalAveragePooling2D()(input_layer)
    fully_connected1 = Dense(units=64, activation='relu')(global_average_pool)  # Two fully connected layers to learn weights
    fully_connected2 = Dense(units=64, activation='relu')(fully_connected1)
    weight_layer = Dense(units=3, activation='linear')(fully_connected2)  # Learning weights
    reshape_layer = Reshape((3, 1))(weight_layer)
    element_wise_multiplication = Multiply()([input_layer, reshape_layer])
    flatten_layer = Flatten()(element_wise_multiplication)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model