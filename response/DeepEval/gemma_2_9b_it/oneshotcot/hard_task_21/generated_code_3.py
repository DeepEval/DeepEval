import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Depthwise Separable Convolutions for each channel group
    x1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    x2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    x3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    # Concatenate outputs from each channel group
    main_path_output = Concatenate(axis=2)([x1, x2, x3])

    # Branch Path
    branch_path = Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Add Main and Branch Path Outputs
    output = main_path_output + branch_path

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model