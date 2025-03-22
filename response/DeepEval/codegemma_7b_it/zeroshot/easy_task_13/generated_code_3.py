import tensorflow as tf
from tensorflow.keras import layers

def dl_model():

    input_layer = layers.Input(shape=(28, 28, 1))

    # Convolutional layers with dropout
    conv1 = layers.Conv2D(32, (1, 1), padding='same')(input_layer)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Dropout(0.2)(conv1)

    conv2 = layers.Conv2D(32, (1, 1), padding='same')(conv1)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Dropout(0.2)(conv2)

    conv3 = layers.Conv2D(32, (3, 1))(conv2)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.2)(conv3)

    conv4 = layers.Conv2D(32, (1, 3))(conv3)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Dropout(0.2)(conv4)

    # Restore channel count
    conv5 = layers.Conv2D(1, (1, 1), padding='same')(conv4)
    conv5 = layers.Activation('relu')(conv5)

    # Combine processed features with input
    output = layers.Add()([conv5, input_layer])

    # Flatten and fully connected layer
    output = layers.Flatten()(output)
    output = layers.Dense(10)(output)
    output = layers.Activation('softmax')(output)

    model = tf.keras.models.Model(inputs=input_layer, outputs=output)

    return model