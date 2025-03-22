from tensorflow.keras import layers, models

def dl_model():

    # Define the input layer
    input_img = layers.Input(shape=(32, 32, 3))

    # First block: Dual-path architecture
    x = layers.Conv2D(64, (3, 3), padding='same')(input_img)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)

    # Branch path
    branch = layers.Conv2D(64, (1, 1), padding='same')(input_img)

    # Combine main and branch paths
    x = layers.add([x, branch])
    x = layers.Activation('relu')(x)

    # Second block: Grouped feature extraction
    x = layers.Reshape((-1, 64))(x)
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=1))(x)

    # Extract features using depthwise separable convolutional layers
    group_1 = layers.Lambda(lambda x: layers.SeparableConv2D(32, (1, 1), padding='same', depthwise_initializer='he_normal')(x))(x[0])
    group_2 = layers.Lambda(lambda x: layers.SeparableConv2D(32, (3, 3), padding='same', depthwise_initializer='he_normal')(x))(x[1])
    group_3 = layers.Lambda(lambda x: layers.SeparableConv2D(32, (5, 5), padding='same', depthwise_initializer='he_normal')(x))(x[2])

    # Concatenate outputs from different groups
    concat = layers.concatenate([group_1, group_2, group_3])

    # Fully connected layers for classification
    x = layers.Dense(128, activation='relu')(concat)
    output = layers.Dense(10, activation='softmax')(x)

    # Create the model
    model = models.Model(inputs=input_img, outputs=output)

    return model