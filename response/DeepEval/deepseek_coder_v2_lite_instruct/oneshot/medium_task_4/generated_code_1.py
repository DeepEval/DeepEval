import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Path 1
    def path1_block(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2)(conv2)
        return avg_pool

    path1_output = path1_block(input_layer)
    path1_output = path1_block(path1_output)

    # Path 2
    conv_path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    avg_pool_path2 = AveragePooling2D(pool_size=(2, 2), strides=2)(conv_path2)

    # Concatenate Path1 and Path2 outputs
    concatenated_output = Concatenate()([path1_output, avg_pool_path2])

    # Batch Normalization
    batch_norm_output = BatchNormalization()(concatenated_output)

    # Flatten the output
    flatten_layer = Flatten()(batch_norm_output)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model