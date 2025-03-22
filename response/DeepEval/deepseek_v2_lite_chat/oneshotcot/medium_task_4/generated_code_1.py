import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Path1
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn1 = BatchNormalization()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bn1)
        return pool1

    def block2(input_tensor):
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        bn2 = BatchNormalization()(conv2)
        pool2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(bn2)
        return pool2

    path1_output = block1(input_layer)
    path1_output = block2(path1_output)

    # Path2
    conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    bn3 = BatchNormalization()(conv3)
    flatten_layer = Flatten()(bn3)

    # Combine the outputs of both paths
    concat_layer = Concatenate()([path1_output, flatten_layer])

    dense1 = Dense(units=128, activation='relu')(concat_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model