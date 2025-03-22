from tensorflow.keras import layers, models

def dl_model():

    # Input layer
    inputs = layers.Input(shape=(32, 32, 3))

    # Main path
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    x = layers.Activation('relu')(x)
    pool_main = layers.GlobalAveragePooling2D()(x)
    fc_main = layers.Dense(units=32)(pool_main)
    fc_main = layers.Activation('relu')(fc_main)
    fc_main = layers.Dense(units=32)(fc_main)
    fc_main = layers.Activation('relu')(fc_main)
    fc_main = layers.Dense(units=10)(fc_main)

    # Branch path
    conv_branch = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(inputs)
    conv_branch = layers.Activation('relu')(conv_branch)
    conv_branch = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(conv_branch)
    conv_branch = layers.Activation('relu')(conv_branch)
    pool_branch = layers.GlobalAveragePooling2D()(conv_branch)
    fc_branch = layers.Dense(units=32)(pool_branch)
    fc_branch = layers.Activation('relu')(fc_branch)
    fc_branch = layers.Dense(units=32)(fc_branch)
    fc_branch = layers.Activation('relu')(fc_branch)
    fc_branch = layers.Dense(units=10)(fc_branch)

    # Combine outputs from both paths
    combined = layers.add([fc_main, fc_branch])
    combined = layers.Activation('softmax')(combined)

    # Model definition
    model = models.Model(inputs=inputs, outputs=combined)

    return model