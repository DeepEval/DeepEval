import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Lambda, Concatenate, Permute, Reshape, BatchNormalization, Flatten, Dense
import tensorflow as tf

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [Conv2D(filters=16 // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate()(x)

    # Block 2
    x = Lambda(lambda x: tf.shape(x)[1:3])(x)
    x = Reshape((x, 3, 16 // 3))(x)
    x = Permute((1, 2, 3, 0))(x)
    x = Reshape((32, 32, 3 * 16 // 3))(x)

    # Block 3
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
               depth_multiplier=1)(x)

    # Branch Path
    branch_x = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)
    branch_x = Flatten()(branch_x)

    # Concatenate
    x = Concatenate()([x, branch_x])
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model