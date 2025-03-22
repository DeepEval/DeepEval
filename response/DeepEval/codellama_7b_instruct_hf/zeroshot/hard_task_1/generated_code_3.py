import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    path1 = GlobalAveragePooling2D()(x)
    path2 = GlobalMaxPooling2D()(x)
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path1 = Dense(16, activation='relu')(path1)
    path2 = Dense(16, activation='relu')(path2)
    path1 = Dense(10, activation='softmax')(path1)
    path2 = Dense(10, activation='softmax')(path2)
    channel_weights = Add()([path1, path2])
    channel_weights = Activation('sigmoid')(channel_weights)
    channel_weights = Reshape((1, 1, 10))(channel_weights)
    x = Multiply()([x, channel_weights])

    # Block 2
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    # Output
    outputs = Add()([x, x])
    outputs = Activation('softmax')(outputs)

    model = keras.models.Model(inputs=inputs, outputs=outputs)
    return model