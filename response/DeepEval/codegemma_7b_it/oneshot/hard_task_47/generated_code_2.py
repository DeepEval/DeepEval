import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    conv1_1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    conv1_3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    conv1_5x5 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    batch_norm1 = BatchNormalization()(Concatenate()([conv1_1x1, conv1_3x3, conv1_5x5]))

    # Second block
    conv2_1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_1x7 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_7x1 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(batch_norm1)
    conv2_3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(Concatenate()([conv2_1x7, conv2_7x1]))
    batch_norm2 = BatchNormalization()(Concatenate()([conv2_1x1, conv2_3x3, conv2_3x3_2]))

    # Pooling and concatenation
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm2)
    flatten_layer = Flatten()(max_pooling)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model