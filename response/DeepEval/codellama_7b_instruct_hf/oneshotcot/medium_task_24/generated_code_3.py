import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define first branch
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = Dropout(rate=0.2)(branch1)

    # Define second branch
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = Dropout(rate=0.2)(branch2)

    # Define third branch
    branch3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_layer)

    # Concatenate outputs from all branches
    concatenate_layer = Concatenate()([branch1, branch2, branch3])

    # Apply batch normalization and flatten layer
    batch_norm = BatchNormalization()(concatenate_layer)
    flatten_layer = Flatten()(batch_norm)

    # Add dense layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model