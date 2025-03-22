import keras
from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = GlobalAveragePooling2D()(inputs)
    main_path = Dense(64, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Define the branch path
    branch_path = Conv2D(10, kernel_size=3, strides=2, padding='same')(inputs)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Activation('relu')(branch_path)

    # Add the branch path to the main path
    outputs = Concatenate()([main_path, branch_path])

    # Define the final layer
    outputs = Dense(10, activation='softmax')(outputs)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model