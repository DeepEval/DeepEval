import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Add, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    main_path = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path[2])

    # Concatenate outputs from the main path
    main_output = Concatenate()([conv1, conv2, conv3])

    # Branch path
    branch_output = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse main and branch paths
    fused_output = Add()([main_output, branch_output])

    # Flatten the result and add fully connected layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model