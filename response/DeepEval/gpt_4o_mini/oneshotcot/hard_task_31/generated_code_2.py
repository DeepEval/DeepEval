import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Dropout(0.25)(main_path)
    main_path = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)

    # Branch path
    branch_path = input_layer

    # Add both paths
    block1_output = Add()([main_path, branch_path])

    # Second block
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block1_output)

    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_inputs[0])
    path1 = Dropout(0.25)(path1)

    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_inputs[1])
    path2 = Dropout(0.25)(path2)

    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_inputs[2])
    path3 = Dropout(0.25)(path3)

    # Concatenate the outputs of the three paths
    block2_output = Concatenate()([path1, path2, path3])

    # Flatten and fully connected layer for output
    flatten_layer = Flatten()(block2_output)
    dense_layer = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model