import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape, Permute

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda tensor: tf.split(tensor, 3, axis=3))(input_layer)
    x = [Conv2D(filters=input_layer.shape[-1]//3, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(i) for i in x]
    x = Concatenate(axis=3)(x)

    # Block 2
    x = Lambda(lambda tensor: tf.split(tensor, 3, axis=3))(x)
    x = [Reshape((tensor.shape[1], tensor.shape[2], 3, tensor.shape[3]//3))(i) for i in x]
    x = [Permute((2, 3, 1, 4))(i) for i in x]
    x = [Reshape((tensor.shape[1], tensor.shape[2], tensor.shape[3]*3))(i) for i in x]
    x = Concatenate(axis=3)(x)

    # Block 3
    x = Conv2D(filters=input_layer.shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    shortcut = input_layer
    x = keras.layers.Add()([x, shortcut])

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model