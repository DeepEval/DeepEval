import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1 = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block1)
    block1 = Concatenate()(block1)

    # Block 2
    block2 = Lambda(lambda x: tf.shape(x))(block1)
    block2 = Lambda(lambda x: tf.reshape(x, (x[0], x[1], x[2], 3)))(block2)
    block2 = Permute((2, 3, 1, 0))(block2)
    block2 = Lambda(lambda x: tf.reshape(x, (x[0], x[1], x[2] * x[3], 1)))(block2)
    block2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = Concatenate()(block2)

    # Block 3
    block3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)

    # Branch path
    branch_path = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(input_layer)
    branch_path = Flatten()(branch_path)

    # Concatenate main and branch path
    concatenated = Concatenate()([block1, block2, block3, branch_path])

    # Batch normalization
    batch_norm = BatchNormalization()(concatenated)

    # Dense layers
    dense1 = Dense(units=128, activation='relu')(batch_norm)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model