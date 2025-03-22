import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    x = [Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(xi) for xi in x]
    x = Concatenate(axis=3)(x) 

    # Block 2
    shape_layer = Lambda(lambda x: tf.shape(x))
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(shape_layer(x)[0])
    x = [Reshape((x[0], x[1], 3, x[2])) for x in zip(x[0], x[1], x[2])] # reshape into (height, width, groups, channels_per_group)
    x = [Permute((2, 3, 1, 4))(xi) for xi in x] # swap the third and fourth dimensions
    x = [Reshape((x[0], x[1], x[2])) for x in zip(x[0], x[1], x[2])] # reshape back to original shape

    # Block 3
    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    shortcut = input_layer
    x = keras.layers.Add()([x, shortcut])

    # Flatten and Dense Layers
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model