import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)

    # Branch 2
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)

    # Branch 3
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input_layer)

    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model