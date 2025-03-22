import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Block 1: Splitting the input into three groups
    x = layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Convolutional layers in Block 1
    x = layers.Conv2D(32, (1, 1), activation='relu')(x[0])
    x = layers.Conv2D(32, (3, 3), activation='relu')(x[1])
    x = layers.Conv2D(32, (1, 1), activation='relu')(x[2])
    x = layers.Concatenate(axis=-1)(x)

    # Transition Convolution layer
    x = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu')(x)

    # Block 2: Global max pooling and channel-matching weights
    x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)

    # Main path output
    main_path_output = layers.Dense(10, activation='softmax')(x)

    # Branch output
    branch_output = layers.Dense(10, activation='softmax')(input_layer)

    # Adding main path and branch outputs
    output = layers.Add()([main_path_output, branch_output])

    # Final output
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    # Compiling the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model