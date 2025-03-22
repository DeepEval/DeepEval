from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Reshape, Dense, concatenate, multiply, Dropout

def dl_model():

    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block: Feature extraction through parallel branches
    branch_1x1 = Conv2D(64, (1, 1), padding='same')(inputs)
    branch_3x3 = Conv2D(64, (3, 3), padding='same')(inputs)
    branch_5x5 = Conv2D(64, (5, 5), padding='same')(inputs)
    branch_pool = MaxPooling2D((2, 2), padding='same')(inputs)

    # Concatenate branch outputs
    merge = concatenate([branch_1x1, branch_3x3, branch_5x5, branch_pool])

    # Second block: Dimensionality reduction and weight generation
    gap = GlobalAveragePooling2D()(merge)
    fc1 = Dense(64, activation='relu')(gap)
    fc2 = Dense(64, activation='relu')(fc1)
    weights = Dense(64, activation='sigmoid')(fc2)

    # Reshape weights to match input shape and perform element-wise multiplication
    weights = Reshape((1, 1, 64))(weights)
    multiply_output = multiply([inputs, weights])

    # Final fully connected layer for classification
    outputs = Dense(10, activation='softmax')(multiply_output)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model