import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First branch (3x3 convolutional layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1)

    # Second branch (5x5 convolutional layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(branch2)

    # Addition of the two branches
    added = Add()([branch1, branch2])

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(added)

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Model construction
    model = Model(inputs=input_layer, outputs=dense2)

    return model