import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input
    inputs = Input(shape=(28, 28, 1))

    # First Block: Main path and branch path
    # Main Path
    x_main = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x_main = Conv2D(1, (3, 3), padding='same', activation='relu')(x_main)

    # Branch Path
    x_branch = inputs  # Direct connection to the input

    # Combine main path and branch path
    x_block1 = Add()([x_main, x_branch])

    # Second Block: Multi-scale pooling
    # Max Pooling with different scales
    pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x_block1)
    pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x_block1)
    pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x_block1)

    # Flatten the pooled features
    flat_1x1 = Flatten()(pool_1x1)
    flat_2x2 = Flatten()(pool_2x2)
    flat_4x4 = Flatten()(pool_4x4)

    # Concatenate flattened features
    x_block2 = Concatenate()([flat_1x1, flat_2x2, flat_4x4])

    # Fully connected layers
    fc1 = Dense(128, activation='relu')(x_block2)
    outputs = Dense(10, activation='softmax')(fc1)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model