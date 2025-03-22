import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(input_layer)
    block1_output = Concatenate()([conv1x1, conv3x3, conv5x5, maxpool])

    # Second block
    gap = GlobalAveragePooling2D()(block1_output)
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    weights = Reshape((1, 1, 32))(dense2)
    weighted_feature_map = Multiply()([weights, block1_output])
    block2_output = GlobalAveragePooling2D()(weighted_feature_map)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(block2_output)

    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model