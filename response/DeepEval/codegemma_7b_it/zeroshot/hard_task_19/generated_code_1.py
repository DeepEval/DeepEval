import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dense, Reshape, multiply
from tensorflow.keras.models import Model

def dl_model():

    # Define input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    main_output = MaxPooling2D()(x)

    # Branch path
    branch_output = GlobalAveragePooling2D()(main_output)
    branch_output = Dense(128, activation='relu')(branch_output)
    branch_output = Dense(main_output.shape[1] * main_output.shape[2] * main_output.shape[3], activation='relu')(branch_output)
    branch_output = Reshape((main_output.shape[1], main_output.shape[2], main_output.shape[3]))(branch_output)
    branch_output = multiply([branch_output, main_output])

    # Combine outputs
    combined_output = concatenate([main_output, branch_output])

    # Classification layers
    combined_output = Dense(256, activation='relu')(combined_output)
    outputs = Dense(10, activation='softmax')(combined_output)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model