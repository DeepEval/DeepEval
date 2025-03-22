import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add, Flatten
from tensorflow.keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images dimensions
    num_classes = 10           # CIFAR-10 has 10 classes

    # Input layer
    inputs = Input(shape=input_shape)

    # Main Path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Branch Path (Squeeze-and-Excitation block)
    branch = GlobalAveragePooling2D()(inputs)
    branch = Dense(64, activation='relu')(branch)
    branch = Dense(128, activation='sigmoid')(branch)
    branch = tf.keras.layers.Reshape((1, 1, 128))(branch)

    # Multiply input with channel weights
    scaled_inputs = Multiply()([x, branch])

    # Combine main path and scaled branch output
    combined = Add()([x, scaled_inputs])

    # Flatten and add classification layers
    flattened = Flatten()(combined)
    dense1 = Dense(256, activation='relu')(flattened)
    dense2 = Dense(128, activation='relu')(dense1)
    outputs = Dense(num_classes, activation='softmax')(dense2)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)

    return model