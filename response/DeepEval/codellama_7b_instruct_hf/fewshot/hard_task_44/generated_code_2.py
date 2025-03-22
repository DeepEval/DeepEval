import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=3))(input_layer)
    x = Conv2D(32, (1, 1), activation='relu')(x[0])
    x = Conv2D(64, (3, 3), activation='relu')(x[1])
    x = Conv2D(128, (5, 5), activation='relu')(x[2])
    x = Dropout(0.2)(x)
    x = Concatenate()([x[0], x[1], x[2]])

    # Block 2
    x = Lambda(lambda x: tf.split(value=x, num_or_size_splits=4, axis=3))(x)
    x = Conv2D(32, (1, 1), activation='relu')(x[0])
    x = Conv2D(64, (1, 1), activation='relu')(x[1])
    x = MaxPooling2D((3, 3), strides=(2, 2))(x[2])
    x = Conv2D(128, (5, 5), activation='relu')(x[3])
    x = Dropout(0.2)(x)
    x = Concatenate()([x[0], x[1], x[2], x[3]])

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model