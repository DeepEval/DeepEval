from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Flatten, Dense, Permute

def dl_model():
    # Primary path
    x = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = DepthwiseConv2D((3, 3), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Branch path
    y = Input(shape=(28, 28, 1))
    y = DepthwiseConv2D((3, 3), activation='relu')(y)
    y = Conv2D(64, (1, 1), activation='relu')(y)
    y = BatchNormalization()(y)
    y = Flatten()(y)

    # Concatenate features
    x = Concatenate()([x, y])

    # Block 2
    x = Permute((3, 1, 2))(x)
    x = Reshape((-1, 4))(x)
    x = Permute((3, 1, 2))(x)
    x = Reshape((-1, 64, 1))(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=[x, y], outputs=x)
    return model