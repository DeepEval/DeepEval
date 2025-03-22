import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # First branch
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)
    branch1 = BatchNormalization()(branch1)

    # Second branch
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.2)(branch2)
    branch2 = BatchNormalization()(branch2)

    # Third branch
    branch3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = Dropout(rate=0.2)(branch3)
    branch3 = BatchNormalization()(branch3)

    # Concatenate branches
    output_layer = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=1024, activation='relu')(output_layer)
    output_layer = Dropout(rate=0.5)(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model