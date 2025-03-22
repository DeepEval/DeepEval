import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Lambda, tf
from keras.layers import BatchNormalization, Flatten, Dense

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_main = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_main)
    x_main = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_main)

    # Branch Path
    x_branch = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine Paths
    x = keras.layers.add([x_main, x_branch])

    # Block 2
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(x)  

    x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    x2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    x3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    x = Concatenate()([x1, x2, x3])
    
    # Final Layers
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model