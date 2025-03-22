import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    max_pooling_layers = []
    for scale in [1, 2, 4]:
        max_pooling_layers.append(MaxPooling2D(pool_size=(scale, scale), strides=(scale, scale), padding='same'))
    max_pooling_outputs = []
    for layer in max_pooling_layers:
        output = layer(input_layer)
        max_pooling_outputs.append(Flatten()(output))
    dropout_layer = Dropout(0.2)(Concatenate()(max_pooling_outputs))

    # Second block
    separable_conv_layers = []
    for kernel_size in [1, 3, 5, 7]:
        separable_conv_layers.append(Lambda(lambda x: tf.split(x, 4, axis=-1))(input_layer))
        output = Concatenate()(separable_conv_layers)
        output = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=kernel_size, strides=(1, 1), padding='same')(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.Activation('relu')(output)
        output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(output)
        output = tf.keras.layers.Concatenate()(output)

    # Flatten and add fully connected layers
    flatten_layer = Flatten()(output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model