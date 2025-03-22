import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.initializers import glorot_uniform

def depthwise_conv2d_wrapper(inputs, kernel_size, strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
    return DepthwiseConv2D(kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs)

def pointwise_conv2d_wrapper(inputs, filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
    return Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(inputs)

def create_model():
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(inputs)
    
    path1 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 32, kernel_size=(1, 1), strides=(1, 1), padding='same'))(split_input[0])
    path2 = Lambda(lambda x: depthwise_conv2d_wrapper(x, kernel_size=(3, 3), strides=(1, 1), padding='same'))(split_input[1])
    path3 = Lambda(lambda x: depthwise_conv2d_wrapper(x, kernel_size=(5, 5), strides=(1, 1), padding='same'))(split_input[2])

    concat = Lambda(lambda x: tf.concat(x, axis=-1))([path1, path2, path3])

    # Block 2
    branch1 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 32, kernel_size=(1, 1), strides=(1, 1), padding='same'))(concat)
    branch2 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 64, kernel_size=(1, 1), strides=(1, 1), padding='same'))(concat)
    branch2 = Lambda(lambda x: depthwise_conv2d_wrapper(x, kernel_size=(3, 3), strides=(1, 1), padding='same'))(branch2)
    branch3 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 64, kernel_size=(1, 1), strides=(1, 1), padding='same'))(concat)
    branch3 = Lambda(lambda x: depthwise_conv2d_wrapper(x, kernel_size=(3, 3), strides=(1, 1), padding='same'))(branch3)
    branch4 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 64, kernel_size=(3, 3), strides=(1, 1), padding='same'))(concat)
    branch4 = Lambda(lambda x: MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))(branch4)
    branch4 = Lambda(lambda x: pointwise_conv2d_wrapper(x, 64, kernel_size=(1, 1), strides=(1, 1), padding='same'))(branch4)

    concat_2 = Lambda(lambda x: tf.concat(x, axis=-1))([branch1, branch2, branch3, branch4])

    # Output layer
    flatten = Flatten()(concat_2)
    outputs = Dense(10, activation='softmax')(flatten)

    model = keras.Model(inputs=inputs, outputs=outputs)
    
    return model

model = create_model()