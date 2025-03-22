import keras
from keras.layers import Input, Conv2D, Concatenate, Add

def dl_model():     

    input_layer = Input(shape=(32, 32, 64))
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
    output_tensor_main = Concatenate()([conv1, conv3])

    branch_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Match channel dimensions of main and branch paths
    output_tensor_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_layer)

    combined_layer = Add()([output_tensor_main, output_tensor_branch])

    bath_norm = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(combined_layer)
    bath_norm = keras.layers.BatchNormalization()(bath_norm)
    flatten_layer = keras.layers.Flatten()(bath_norm)
    dense1 = keras.layers.Dense(units=128, activation='relu')(flatten_layer)
    output_layer = keras.layers.Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model