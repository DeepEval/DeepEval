import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block1_output = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(conv)
    block1_output = [Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x) for x in block1_output]
    block1_output = Concatenate()([x for x in block1_output])
    
    # Block 2
    block2_output = Lambda(lambda x: K.reshape(x, (-1, 8, 8, 3, 16)))(block1_output)
    block2_output = Lambda(lambda x: tf.transpose(x, perm=(0, 1, 2, 4, 3)))(block2_output)
    block2_output = Lambda(lambda x: K.reshape(x, (-1, 8, 8, 48)))(block2_output)
    
    # Block 3
    block3_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', groups=48)(block2_output)

    # Branch path
    average_pooling = Lambda(lambda x: K.mean(x, axis=(1, 2), keepdims=True))(input_layer)
    
    # Concatenate outputs from main path and branch path
    concatenated_output = Concatenate()([block3_output, average_pooling])

    # Batch normalization and flatten
    bath_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(bath_norm)

    # Dense layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model