import keras
from keras.layers import Input, Conv2D, Lambda, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Main Path

    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    x = [Conv2D(filters=int(keras.backend.int_shape(input_layer)[3] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate(axis=3)(x)

    # Block 2
    x = Lambda(lambda x: tf.shape(x)[1:3])(x)
    x = Reshape((keras.backend.int_shape(x)[1], keras.backend.int_shape(x)[2], 3, int(keras.backend.int_shape(x)[3] / 3)))(x)
    x = Permute((1, 2, 3, 4))(x)
    x = Reshape((keras.backend.int_shape(x)[1], keras.backend.int_shape(x)[2], keras.backend.int_shape(x)[3]))(x)

    # Block 3
    x = Conv2D(filters=int(keras.backend.int_shape(x)[3]), kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depth_wise=True)(x)

    # Branch Path
    branch_output = AveragePooling2D(pool_size=(8, 8))(input_layer)
    branch_output = Flatten()(branch_output)

    # Concatenate Outputs
    concatenated_output = Concatenate()([x, branch_output])

    # Fully Connected Layer
    output_layer = Dense(units=10, activation='softmax')(concatenated_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model