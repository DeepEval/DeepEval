import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Permute, Reshape, Concatenate
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1_input = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    block1_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1_input)
    block1_output = Concatenate()(block1_output)

    # Block 2
    block2_input = Reshape(target_shape=(32, 32, 3))(block1_output)
    block2_output = Permute(permutation=[0, 2, 1, 3])(block2_input)
    block2_output = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2_output)
    block2_output = Reshape(target_shape=(32, 32, 3))(block2_output)

    # Block 3
    block3_input = Reshape(target_shape=(32, 32, 3))(block2_output)
    block3_output = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block3_input)

    # Addition layer
    addition_layer = Add()([block1_output, block3_output])

    # Flatten layer
    flatten_layer = Flatten()(addition_layer)

    # Dense layer
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Model
    model = Model(inputs=input_layer, outputs=dense_layer)

    return model