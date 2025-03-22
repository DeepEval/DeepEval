import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.initializers import glorot_uniform

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    group_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group_2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group_3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Feature extraction for each group
    conv_1_1x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=glorot_uniform(seed=None))(group_1)
    conv_3_3x3 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=glorot_uniform(seed=None))(group_2)
    conv_5_5x5 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', use_bias=False, kernel_initializer=glorot_uniform(seed=None))(group_3)

    # Batch normalization and activation
    batch_norm_1_1x1 = BatchNormalization()(conv_1_1x1)
    batch_norm_3_3x3 = BatchNormalization()(conv_3_3x3)
    batch_norm_5_5x5 = BatchNormalization()(conv_5_5x5)

    # Max pooling and concatenation
    max_pool_1_1x1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm_1_1x1)
    max_pool_3_3x3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm_3_3x3)
    max_pool_5_5x5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(batch_norm_5_5x5)

    concat = Concatenate()([max_pool_1_1x1, max_pool_3_3x3, max_pool_5_5x5])

    # Flatten and fully connected layer
    flatten = Flatten()(concat)
    dense = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=dense)

    return model