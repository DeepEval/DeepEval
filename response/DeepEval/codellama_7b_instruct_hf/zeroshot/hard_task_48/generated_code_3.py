from keras.layers import Input, Lambda, concatenate, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x[0])
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x[1])
    x3 = Conv2D(128, (5, 5), padding='same', activation='relu')(x[2])
    x = concatenate([x1, x2, x3])
    x = BatchNormalization()(x)

    # Block 2
    y1 = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    y2 = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    y3 = Conv2D(128, (5, 5), padding='same', activation='relu')(x)
    y4 = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    y = concatenate([y1, y2, y3, y4])
    y = BatchNormalization()(y)

    # Flatten layer
    flattened_y = Flatten()(y)

    # Fully connected layer
    fc_layer = Dense(10, activation='softmax')(flattened_y)

    # Create model
    model = Model(inputs=input_layer, outputs=fc_layer)

    return model