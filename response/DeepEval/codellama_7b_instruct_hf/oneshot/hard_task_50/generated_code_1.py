from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Reshape

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    x = input_layer
    x = Conv2D(32, (1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Conv2D(128, (4, 4), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)

    # Second block
    x = Lambda(lambda x: tf.split(x, 4, axis=-1))(x)
    x = [Conv2D(32, (1, 1), padding='same', activation='relu')(x[i]) for i in range(4)]
    x = [Conv2D(32, (3, 3), padding='same', activation='relu')(x[i]) for i in range(4)]
    x = [Conv2D(32, (5, 5), padding='same', activation='relu')(x[i]) for i in range(4)]
    x = [Conv2D(32, (7, 7), padding='same', activation='relu')(x[i]) for i in range(4)]
    x = [MaxPooling2D(pool_size=(2, 2), strides=2)(x[i]) for i in range(4)]
    x = [Flatten()(x[i]) for i in range(4)]
    x = [Dropout(0.2)(x[i]) for i in range(4)]
    x = Reshape((4, -1))(x)

    # Fully connected layers
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model