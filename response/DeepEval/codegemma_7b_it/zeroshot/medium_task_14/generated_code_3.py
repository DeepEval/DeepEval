import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Activation, Input, concatenate, Conv2DTranspose, Reshape, Dense, Flatten, Dropout

def dl_model():
    inputs = Input(shape=(32, 32, 3))

    # Block 1
    block1 = Conv2D(64, (3, 3), padding='same')(inputs)
    block1 = BatchNormalization()(block1)
    block1 = Activation('relu')(block1)

    # Block 2
    block2 = Conv2D(128, (3, 3), padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('relu')(block2)

    # Block 3
    block3 = Conv2D(256, (3, 3), padding='same')(block2)
    block3 = BatchNormalization()(block3)
    block3 = Activation('relu')(block3)

    # Parallel branch
    shortcut = Conv2D(256, (1, 1), padding='same')(inputs)

    # Output paths
    path1 = Conv2D(128, (3, 3), padding='same')(block1)
    path1 = BatchNormalization()(path1)
    path1 = Activation('relu')(path1)

    path2 = Conv2D(128, (3, 3), padding='same')(block2)
    path2 = BatchNormalization()(path2)
    path2 = Activation('relu')(path2)

    path3 = Conv2D(128, (3, 3), padding='same')(block3)
    path3 = BatchNormalization()(path3)
    path3 = Activation('relu')(path3)

    # Aggregated output
    outputs = concatenate([path1, path2, path3, shortcut])

    # Fully connected layers
    outputs = Flatten()(outputs)
    outputs = Dense(512, activation='relu')(outputs)
    outputs = Dropout(0.2)(outputs)
    outputs = Dense(10, activation='softmax')(outputs)

    # Model construction
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model