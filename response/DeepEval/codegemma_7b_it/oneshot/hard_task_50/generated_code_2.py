import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def max_pool_block(input_tensor):
        max_pool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        max_pool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        max_pool_concat = Concatenate()([max_pool1, max_pool2, max_pool3])
        max_pool_dropout = Dropout(0.2)(max_pool_concat)
        max_pool_flatten = Flatten()(max_pool_dropout)
        return max_pool_flatten

    max_pool_output = max_pool_block(input_tensor)

    # Reshape output for second block
    reshape = keras.layers.Reshape((1, 1, -1))(max_pool_output)

    # Second block
    def separable_conv_block(input_tensor):
        split = Lambda(lambda x: tf.split(x, 4, axis=-1))(input_tensor)
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split[2])
        conv4 = Conv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split[3])
        concat = Concatenate()([conv1, conv2, conv3, conv4])
        return concat

    separable_conv_output = separable_conv_block(reshape)

    # Flatten and fully connected layer for classification
    flatten = Flatten()(separable_conv_output)
    dense = Dense(units=10, activation='softmax')(flatten)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=dense)

    return model