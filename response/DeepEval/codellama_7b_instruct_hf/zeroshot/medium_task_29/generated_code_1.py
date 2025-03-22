import tensorflow as tf
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(32, 32, 3))

    # Convolutional layers
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    # Flatten layers
    flatten = layers.Flatten()(pool3)

    # Dense layers
    dense1 = layers.Dense(128, activation='relu')(flatten)
    output = layers.Dense(10, activation='softmax')(dense1)

    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model