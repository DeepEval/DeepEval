import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Dual-path block
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
    branch_path = input_layer

    # Concatenate main and branch paths
    output = Concatenate()([main_path, branch_path])

    # Split input into three groups
    output = Lambda(lambda x: tf.split(x, 3, axis=3))(output)

    # Depthwise separable convolutional layers
    output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output[0])
    output = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(output[1])
    output = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(output[2])

    # Concatenate outputs from depthwise separable convolutional layers
    output = Concatenate()(output)

    # Batch normalization and flatten
    output = BatchNormalization()(output)
    output = Flatten()(output)

    # Fully connected layers
    output = Dense(units=128, activation='relu')(output)
    output = Dense(units=64, activation='relu')(output)
    output = Dense(units=10, activation='softmax')(output)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model