import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model
import tensorflow as tf

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Initial convolution and splitting of channels
    conv1 = Conv2D(32, (1, 1), padding='same')(input_layer)
    split1, split2 = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv1)
    
    # Operations for the first group
    conv2 = Conv2D(32, (1, 1), padding='same', activation='relu')(split1)
    dwconv = keras.layers.DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2)(split1)
    conv3 = Conv2D(64, (1, 1), padding='same', activation='relu')(dwconv)
    
    # Operations for the second group
    split1, split2 = Lambda(lambda x: tf.split(x, 2, axis=-1))(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(split1)
    dwconv = keras.layers.DepthwiseConv2D((3, 3), padding='same', depth_multiplier=2)(split1)
    conv3 = Conv2D(64, (1, 1), padding='same', activation='relu')(dwconv)
    
    # Concatenate and batch normalize
    concat = Concatenate()([conv2, conv3])
    batchnorm = BatchNormalization()(concat)
    
    # Flatten and dense layers for Block 1
    flatten = Flatten()(batchnorm)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Block 2: Channel shuffling and reshaping
    input_shape = tf.shape(concat)
    batch_size = tf.cast(input_shape[0], dtype=tf.int32)
    height, width = input_shape[1], input_shape[2]
    
    # Reshape to 4 groups
    reshape1 = Lambda(lambda x: tf.reshape(x, (batch_size, height, width, 2)))(concat)
    reshape2 = Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(concat)
    reshape3 = Lambda(lambda x: tf.reshape(x, (batch_size, height * width * 2, 2)))((concat))
    
    # Flatten and dense layers for Block 2
    flatten = Flatten()(reshape3)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model