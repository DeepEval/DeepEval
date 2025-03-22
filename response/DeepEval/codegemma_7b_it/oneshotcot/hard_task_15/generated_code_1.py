import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, BatchNormalization, Concatenate, Flatten, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv)
    gap = GlobalAveragePooling2D()(max_pooling)

    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=input_layer.shape[2], activation='sigmoid')(dense1)
    reshaped_weights = Reshape((input_layer.shape[2], input_layer.shape[3], input_layer.shape[4]))(dense2)
    weighted_feature_map = multiply([input_layer, reshaped_weights])

    # Branch path
    branch_path = input_layer

    # Combine paths
    combined = Concatenate()([weighted_feature_map, branch_path])

    # Fully connected layers
    flatten_layer = Flatten()(combined)
    dense3 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense3)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model