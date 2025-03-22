import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.initializers import glorot_uniform
from tensorflow.keras import layers

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    # First Block
    def split_group(input_tensor):
        split_input = tf.split(input_tensor, num_or_size_splits=3, axis=-1)
        conv1 = layers.Conv2D(32, (1, 1), padding="same", kernel_initializer=glorot_uniform())(split_input[0])
        conv2 = layers.Conv2D(32, (3, 3), padding="same", kernel_initializer=glorot_uniform())(split_input[1])
        conv3 = layers.Conv2D(32, (5, 5), padding="same", kernel_initializer=glorot_uniform())(split_input[2])
        return [conv1, conv2, conv3]

    group_output = Lambda(split_group)(input_layer)
    concat_layer = Concatenate(axis=-1)(group_output)

    # Second Block
    def feature_extraction(input_tensor):
        conv1 = layers.Conv2D(64, (1, 1), padding="same", kernel_initializer=glorot_uniform())(input_tensor)
        conv2 = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=glorot_uniform())(input_tensor)
        conv3 = layers.Conv2D(64, (3, 3), padding="same", kernel_initializer=glorot_uniform())(input_tensor)
        conv4 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(input_tensor)
        conv5 = layers.Conv2D(64, (1, 1), padding="same", kernel_initializer=glorot_uniform())(conv4)
        return [conv1, conv2, conv3, conv4, conv5]

    feature_output = Lambda(feature_extraction)(concat_layer)
    concat_feature = Concatenate(axis=-1)(feature_output)

    # Output Layer
    flatten_layer = Flatten()(concat_feature)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model