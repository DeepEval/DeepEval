import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dense, Add, Activation
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    # Path 1: Global average pooling
    path1 = GlobalAveragePooling2D()(input_layer)
    path1 = Dense(64, activation='relu')(path1)
    path1 = Dense(32, activation='relu')(path1)

    # Path 2: Global max pooling
    path2 = GlobalMaxPooling2D()(input_layer)
    path2 = Dense(64, activation='relu')(path2)
    path2 = Dense(32, activation='relu')(path2)

    # Add and activate
    path_output = Add()([path1, path2])
    path_output = Activation('relu')(path_output)

    # Block 2
    # Average pooling
    block2_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path_output)
    block2_pool = Flatten()(block2_pool)
    block2_pool = Dense(128, activation='relu')(block2_pool)
    block2_pool = Dense(64, activation='relu')(block2_pool)

    # Max pooling
    block2_max = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(path_output)
    block2_max = Flatten()(block2_max)
    block2_max = Dense(128, activation='relu')(block2_max)
    block2_max = Dense(64, activation='relu')(block2_max)

    # Concatenate and add
    block2_output = Add()([block2_pool, block2_max])
    block2_output = Activation('relu')(block2_output)

    # Final classification
    output_layer = Dense(10, activation='softmax')(block2_output)

    # Create and return model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model