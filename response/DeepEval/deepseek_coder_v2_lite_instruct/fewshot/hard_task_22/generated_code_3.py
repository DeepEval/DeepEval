import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv1_3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    conv1_5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    main_path = Concatenate()([conv1_1, conv1_3, conv1_5])

    # Branch Path
    conv2_1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = conv2_1

    # Fuse paths
    added_layer = Add()([main_path, branch_path])

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(added_layer)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model