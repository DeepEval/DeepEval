import keras
from keras.layers import Input, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Add, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    gap = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(gap)  # Same channels as input (3 channels)
    dense2 = Dense(units=96, activation='relu')(dense1)  # Same channels as input (3 channels)
    
    # Reshape weights to match input shape
    reshaped_weights = Reshape((1, 1, 96))(dense2)
    weighted_features = keras.layers.multiply([input_layer, reshaped_weights])

    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Combine outputs from Block 1 and Block 2
    combined = Add()([weighted_features, max_pool])

    # Classifying with fully connected layers
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model