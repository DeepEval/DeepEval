import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    def path1(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    def path2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)

    def path3(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        return Concatenate()([conv1x3, conv3x1])

    def path4(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        return Concatenate()([conv1x3, conv3x1])

    # Apply paths to input layer
    path1_output = path1(input_layer)
    path2_output = path2(input_layer)
    path3_output = path3(input_layer)
    path4_output = path4(input_layer)

    # Concatenate outputs of all paths
    concatenated_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model