import keras
from keras.layers import Input, Lambda, Conv2D, Activation, AveragePooling2D, Concatenate, Reshape, Permute, DepthwiseConv2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch Path
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Main Path
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    x = [Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same')(i) for i in x]
    x = [Activation('relu')(i) for i in x]
    x = Concatenate()(x)

    # Block 2
    shape = Lambda(lambda x: tf.shape(x))(x)
    x = Reshape(target_shape=(shape[1], shape[2], 3, int(shape[3]/3)))(x)
    x = Permute(dims=(0, 1, 3, 2))(x)
    x = Reshape(target_shape=(shape[1], shape[2], shape[3]))(x)

    # Block 3
    x = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)

    # Repeat Block 1
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(x)
    x = [Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same')(i) for i in x]
    x = [Activation('relu')(i) for i in x]
    x = Concatenate()(x)

    # Concatenate main path and branch path
    combined = Concatenate()([x, avg_pool])

    # Classification
    flatten = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model