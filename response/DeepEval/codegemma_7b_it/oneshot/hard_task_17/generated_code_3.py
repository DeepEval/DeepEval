import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add
from keras.models import Model

def dl_model():

    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(64, activation='relu')(gap)
    dense2 = Dense(64, activation='relu')(dense1)
    weights = Reshape((1, 1, 64))(dense2)
    weighted_output = Multiply()([input_layer, weights])

    # Block 2
    conv1 = Conv2D(64, (3, 3), padding='same', activation='relu')(weighted_output)
    conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D()(conv2)

    # Branch from Block 1
    branch_output = Reshape((1, 1, 64))(dense2)

    # Fusion
    fused_output = Add()([max_pooling, branch_output])

    # Output layer
    flatten = Flatten()(fused_output)
    dense3 = Dense(64, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense3)

    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)

    return model