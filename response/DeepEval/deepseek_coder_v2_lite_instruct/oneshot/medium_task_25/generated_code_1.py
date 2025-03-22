import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Define the four parallel paths
    def path_1(input_tensor):
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)

    def path_2(input_tensor):
        avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_tensor)
        return Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(avg_pool)

    def path_3(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        return Concatenate()([conv1x3, conv3x1])

    def path_4(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        conv1x3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        conv3x1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        return Concatenate()([conv1x3, conv3x1])

    # Apply paths to the pooled layer
    path1_output = path_1(pool1)
    path2_output = path_2(pool1)
    path3_output = path_3(pool1)
    path4_output = path_4(pool1)

    # Concatenate outputs of all paths
    concat_output = Concatenate()([path1_output, path2_output, path3_output, path4_output])

    # Batch normalization and flatten
    batch_norm = BatchNormalization()(concat_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model