import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Reshape, Permute, DepthwiseConv2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch path
    branch_path = AveragePooling2D(pool_size=(8, 8), strides=(8, 8), padding='same')(input_layer)

    # Main path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    x = [Conv2D(filters=int(keras.backend.shape(input_layer)[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate()(x)

    # Block 2
    x_shape = keras.layers.Lambda(lambda x: keras.backend.shape(x))(x)
    x = Reshape(target_shape=(x_shape[1], x_shape[2], 3, int(x_shape[3] / 3)))(x)
    x = Permute(axes=[0, 1, 3, 2])(x)
    x = Reshape(target_shape=(x_shape[1], x_shape[2], x_shape[3]))(x)

    # Block 3
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Repeat Block 1
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(x)
    x = [Conv2D(filters=int(keras.backend.shape(input_layer)[-1] / 3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate()(x)

    # Concatenate outputs
    x = Concatenate()([x, branch_path])

    # Fully connected layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model