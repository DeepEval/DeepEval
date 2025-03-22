import keras
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    block1 = GlobalAveragePooling2D()(input_layer)
    block1 = Flatten()(block1)
    block1 = Dense(units=64, activation='relu')(block1)
    block1 = Dense(units=64, activation='relu')(block1)

    # Block 2
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)
    block2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(block2)

    # Branch from Block 1
    branch = Concatenate()([block1, block2])
    branch = Dense(units=64, activation='relu')(branch)
    branch = Dense(units=64, activation='relu')(branch)

    # Combine the outputs from the main path and the branch
    output = Concatenate()([block2, branch])

    # Add dropout layer to prevent overfitting
    output = Dropout(0.2)(output)

    # Final classification layer
    output = Dense(units=10, activation='softmax')(output)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output)

    return model