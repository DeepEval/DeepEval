import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def dl_model():
    # Input layer
    input_layer = layers.Input(shape=(28, 28, 1))

    # Convolutional layer
    conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
    # Max pooling layer
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    # Flatten the feature maps
    flatten = layers.Flatten()(pool1)

    # Dense layers
    dense1 = layers.Dense(64, activation='relu')(flatten)
    output = layers.Dense(10, activation='softmax')(dense1)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model