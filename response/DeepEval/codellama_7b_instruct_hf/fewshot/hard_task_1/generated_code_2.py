import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Activation, Add, Multiply

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Parallel processing paths
    path1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    path1 = GlobalAveragePooling2D()(path1)
    path1 = Dense(128, activation='relu')(path1)
    path1 = Dense(64, activation='relu')(path1)

    path2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    path2 = GlobalMaxPooling2D()(path2)
    path2 = Dense(128, activation='relu')(path2)
    path2 = Dense(64, activation='relu')(path2)

    # Channel attention weights
    channel_weights = Dense(32, activation='softmax')(input_layer)

    # Block 2: Extract spatial features
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path1 = Dense(128, activation='relu')(path1)
    path2 = Dense(128, activation='relu')(path2)
    path1 = Dense(64, activation='relu')(path1)
    path2 = Dense(64, activation='relu')(path2)

    # Add and normalize features
    path1 = Add()([path1, path2])
    path1 = Activation('sigmoid')(path1)
    path1 = Multiply()([channel_weights, path1])

    # Block 3: Fully connected layer
    output_layer = Dense(10, activation='softmax')(path1)

    # Create and return model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model