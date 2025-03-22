import keras
from keras.layers import Input, Conv2D, Lambda, Flatten, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))

    # Block 1
    block1_conv = Conv2D(32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    block1_splitted = Lambda(lambda x: tf.split(value=x, num_or_size_splits=2, axis=-1))(block1_conv)
    block1_conv1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(block1_splitted[0])
    block1_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(block1_splitted[1])
    block1_conv3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(block1_conv2)
    block1_concat = Concatenate()([block1_conv1, block1_conv3])

    # Block 2
    block2_shape = Reshape(target_shape=(28, 28, 2, 16))(block1_concat)
    block2_perm = Lambda(lambda x: tf.transpose(x, perm=[0, 1, 3, 2]))(block2_shape)
    block2_reshape = Reshape(target_shape=(28, 28, 16))(block2_perm)
    block2_conv = Conv2D(128, kernel_size=(1, 1), padding='same', activation='relu')(block2_reshape)
    block2_conv1 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(block2_conv)
    block2_conv2 = Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu')(block2_conv1)
    block2_conv3 = Conv2D(64, kernel_size=(1, 1), padding='same', activation='relu')(block2_conv2)
    block2_concat = Concatenate()([block2_conv1, block2_conv3])

    # Flatten and output
    block2_flatten = Flatten()(block2_concat)
    output_layer = Dense(units=10, activation='softmax')(block2_flatten)

    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model