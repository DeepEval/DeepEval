import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf
from keras.datasets import cifar10

def dl_model(): 
    input_layer = Input(shape=(32, 32, 3)) 

    # Block 1: Main path and branch path
    x_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_main = Dropout(0.2)(x_main)
    x_main = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_main) 

    x_branch = input_layer

    x_concat = keras.layers.add([x_main, x_branch])

    # Block 2: Splitting and separable convolutions
    x_split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(x_concat) 
    
    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_split[0])
    x1 = Dropout(0.2)(x1)
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_split[1])
    x2 = Dropout(0.2)(x2)
    x3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x_split[2])
    x3 = Dropout(0.2)(x3)

    x_concat_2 = Concatenate()( [x1, x2, x3] )

    x_flatten = Flatten()(x_concat_2)
    output_layer = Dense(units=10, activation='softmax')(x_flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model