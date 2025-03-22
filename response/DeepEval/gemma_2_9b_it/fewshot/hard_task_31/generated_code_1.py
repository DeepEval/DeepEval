import keras
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Lambda, Concatenate, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Main Path and Branch Path
    x_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x_main = Dropout(0.2)(x_main)
    x_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_main)
    x_branch = input_layer
    x_combined = keras.layers.add([x_main, x_branch])

    # Block 2: Split and Separable Convolutions
    x_split = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(x_combined)
    
    x1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x_split[0])
    x1 = Dropout(0.2)(x1)
    
    x2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x_split[1])
    x2 = Dropout(0.2)(x2)

    x3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x_split[2])
    x3 = Dropout(0.2)(x3)

    x_concat = Concatenate()([x1, x2, x3])

    # Output
    x_flat = Flatten()(x_concat)
    output_layer = Dense(units=10, activation='softmax')(x_flat)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model