import keras
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Permute
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the initial convolutional layer
    initial_conv = Conv2D(32, (3, 3), activation='relu')

    # Define the first sequential block
    block_1 = Lambda(lambda x: tf.split(x, 2, axis=-1))(initial_conv(input_shape))

    # Define the first group
    group_1 = Conv2D(64, (1, 1), activation='relu')(block_1)
    group_1 = Conv2D(128, (3, 3), activation='relu')(group_1)
    group_1 = Conv2D(64, (1, 1), activation='relu')(group_1)

    # Define the second group
    group_2 = block_1

    # Define the concatenation layer
    concat = Lambda(lambda x: tf.concat(x, axis=-1))([group_1, group_2])

    # Define the second sequential block
    block_2 = Lambda(lambda x: tf.split(x, 2, axis=-1))(concat)

    # Define the shuffling layer
    shuffle = Permute((0, 2, 3, 1))(block_2)

    # Define the reshaping layer
    reshape = Lambda(lambda x: tf.reshape(x, (28, 28, 2, 64)))(shuffle)

    # Define the fully connected layer
    fc = Dense(10, activation='softmax')(reshape)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc)

    return model