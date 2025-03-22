import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():  

    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Main Path and Branch Path
    x_main = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_main = Dropout(0.2)(x_main)
    x_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_main) 
    x_branch = input_layer
    x_combined = keras.layers.add([x_main, x_branch])

    # Block 2: Splitting and Separable Convolutions
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x_combined) 
    
    # Group 1
    x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    x1 = Dropout(0.2)(x1)

    # Group 2
    x2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    x2 = Dropout(0.2)(x2)

    # Group 3
    x3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    x3 = Dropout(0.2)(x3)

    x = Concatenate(axis=3)([x1, x2, x3])

    # Flatten and Output
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model