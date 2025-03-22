import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Add, Flatten, Dense, concatenate

def dl_model():     
    input_layer = Input(shape=(32, 32, 3))

    # Splitting the input into three groups along the channel dimension
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Convolution and pooling for each group
    conv_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(channel_splits[0])
    conv_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(channel_splits[1])
    conv_3 = Conv2D(32, (1, 1), padding='same', activation='relu')(channel_splits[2])

    # Main path
    main_path = concatenate([conv_1, conv_2, conv_3])
    pool_1 = MaxPooling2D(pool_size=(2, 2))(main_path)

    # Fusing the main path with the original input
    fused = Add()([input_layer, pool_1])

    # Flattening the fused features and feeding into fully connected layers
    flatten = Flatten()(fused)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model