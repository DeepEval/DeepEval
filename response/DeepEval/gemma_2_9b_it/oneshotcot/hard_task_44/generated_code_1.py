import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, Dropout, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():     
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    # Feature extraction with varying kernels
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Concatenate outputs
    x = Concatenate()([branch1, branch2, branch3])
    x = Dropout(0.25)(x) # Dropout for regularization

    # Block 2
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=1))(x)

    # Branch 1
    branch4_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch4_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[1])
    # Branch 2
    branch5_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[2])
    branch5_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[3])
    # Branch 3
    branch6_1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x[4])
    branch6_2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch6_1)

    # Concatenate outputs
    x = Concatenate()([branch4_1, branch4_2, branch5_1, branch5_2, branch6_2])

    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model