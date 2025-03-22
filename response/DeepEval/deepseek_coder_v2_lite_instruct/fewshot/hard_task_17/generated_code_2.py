import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Add, Flatten
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    def block_1(input_tensor):
        # Global Average Pooling
        gap = GlobalAveragePooling2D()(input_tensor)
        # Two fully connected layers with the same number of channels as the input
        fc1 = Dense(units=32, activation='relu')(gap)
        fc2 = Dense(units=32, activation='relu')(fc1)
        # Reshape to match input shape
        reshaped = Dense(units=32*32*3, activation='sigmoid')(fc2)
        # Multiply with input to produce weighted feature output
        weighted_features = keras.layers.multiply([reshaped, input_tensor])
        return weighted_features

    block1_output = block_1(input_layer)

    # Block 2
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(block1_output)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Branch from Block 1
    branch = block_1(max_pool)

    # Addition of main path and branch
    added = Add()([block1_output, branch])

    # Flatten and two fully connected layers for classification
    flattened = Flatten()(added)
    fc1 = Dense(units=128, activation='relu')(flattened)
    output_layer = Dense(units=10, activation='softmax')(fc1)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model