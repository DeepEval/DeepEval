import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Permute, Reshape, Lambda

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    inputs_groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(conv1)
    conv2 = Conv2D(64, (1, 1), activation='relu')(inputs_groups[0])
    conv3 = Conv2D(128, (3, 3), activation='relu')(inputs_groups[1])
    merge = Concatenate()([conv2, conv3])
    flatten = Flatten()(merge)

    # Block 2
    reshape = Reshape((28, 28, 2, 64))(flatten)
    permute = Permute((2, 1, 3))(reshape)
    reshape_again = Reshape((28, 28, 64))(permute)
    dense = Dense(10, activation='softmax')(reshape_again)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model