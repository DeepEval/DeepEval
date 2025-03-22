import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    x = [Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(xi) for xi in x]
    x = Concatenate(axis=3)(x)

    # Block 2
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    x = [tf.keras.layers.Reshape((x.shape[1], x.shape[2], 3, x.shape[3]//3))(xi) for xi in x]
    x = [tf.transpose(xi, perm=[0, 1, 3, 2]) for xi in x]
    x = [tf.keras.layers.Reshape((x.shape[1], x.shape[2], x.shape[3]*3))(xi) for xi in x]
    x = Concatenate(axis=3)(x)

    # Block 3
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    shortcut = input_layer
    x = x + shortcut  

    # Final layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model