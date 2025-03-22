import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute
from keras.layers import Add

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Splitting and 1x1 Convolutions
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x = [Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), activation='relu')(xi) for xi in x]
    x = Concatenate(axis=3)(x)

    # Block 2: Channel Shuffling
    shape = Lambda(lambda x: tf.shape(x))(x)
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [Reshape((x[0][0], x[0][1], 3, x[0][2] // 3))(xi) for xi in x]
    x = [Permute((1, 2, 4, 3))(xi) for xi in x]
    x = [Reshape((x[0][0], x[0][1], x[0][2] * 3))(xi) for xi in x]
    x = Concatenate(axis=3)(x)

    # Block 3: Depthwise Separable Conv and Residual Connection
    x = Conv2D(filters=x.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_multiplier=1)(x)
    x_skip = Input(shape=input_layer.shape[1:])
    x = Add()([x, x_skip])

    # Final Fully Connected Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=[input_layer, x_skip], outputs=output_layer)

    return model