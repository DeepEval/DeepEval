import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three groups along the channel dimension
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3)[0])(input_layer)
    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3)[1])(input_layer)
    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3)[2])(input_layer)

    # Apply different convolutional kernels to each group
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2)
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group3)

    # Concatenate the outputs from the three groups
    concat = Concatenate()([conv1, conv2, conv3])
    
    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(concat)
    flatten_layer = Flatten()(bath_norm)

    # Two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model