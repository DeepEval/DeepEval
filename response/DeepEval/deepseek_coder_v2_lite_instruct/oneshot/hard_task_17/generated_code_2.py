import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten, Add, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Fully connected layer to generate weights
    weights = GlobalAveragePooling2D()(x)
    weights = Dense(32, activation='relu')(weights)
    weights = Dense(32 * 32 * 3, activation='sigmoid')(weights)
    weights = weights.reshape((weights.shape[0], 32, 32, 3))

    # Weighted feature output
    weighted_feature = Multiply()([weights, input_layer])

    # Block 2
    y = Conv2D(64, (3, 3), activation='relu')(weighted_feature)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D(pool_size=(2, 2))(y)

    # Branch from Block 1
    branch = MaxPooling2D(pool_size=(2, 2))(x)

    # Fuse the main path and the branch
    combined = Add()([y, branch])

    # Flatten and add fully connected layers
    combined_flatten = Flatten()(combined)
    dense1 = Dense(256, activation='relu')(combined_flatten)
    output_layer = Dense(10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model